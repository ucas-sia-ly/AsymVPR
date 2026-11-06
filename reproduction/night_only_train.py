"""Night-query specialist: no day-query replay, pure night-teacher targets.

Use the same original initialization, night snapshot, night exposure count and
optimizer-update budget as night_v1. Day gallery statistics stay frozen because
the deployed gallery still uses the original BoQ teacher feature space.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from reproduction import night_train as shared
from reproduction.asym_model import EfficientViTBoQ
from reproduction.evaluate import atomic_json, load_inference_checkpoint, sha256
from reproduction.monitoring import TrainingLogger
from reproduction.night_data import load_records
from reproduction.night_specialist import (NightOnlyQueries, night_only_objective,
    select_night_best, validate_night_only_config)
from reproduction.run import atomic_torch_save, configure_cuda


def validate_reference_inputs(config):
    """Fail before GPU work if the fixed mixed reference or shared inputs differ."""
    from reproduction.night_finalize import _read_sealed
    summary_path = Path(config['mixed_reference_summary'])
    comparison_path = Path(config['mixed_reference_comparison'])
    for kind, path in [('summary', summary_path), ('comparison', comparison_path)]:
        if sha256(path) != config[f'mixed_reference_{kind}_sha256']:
            raise ValueError(f'Pinned mixed {kind} checksum mismatch')
    summary = json.loads(summary_path.read_text())
    comparison = _read_sealed(comparison_path)
    mixed_config = comparison['training_config']
    for key in ('init_checkpoint', 'input_dir', 'night_teacher_cache', 'base_cache', 'teacher_weights',
                'svox_dev_manifest', 'seed', 'night_batch_size', 'epochs', 'lr_backbone',
                'lr_aggregator', 'min_lr_ratio', 'temperature', 'gamma'):
        if mixed_config[key] != config[key]:
            raise ValueError(f'Controlled night-only experiment differs from mixed reference: {key}')
    old_root = Path(mixed_config['output_dir']).resolve().parent
    new_output = Path(config['output_dir']).resolve()
    if new_output == old_root or old_root in new_output.parents:
        raise ValueError('Night-only outputs must be outside the original mixed experiment')
    mixed_best = Path(mixed_config['output_dir']) / 'best.pt'
    if summary['checkpoint_sha256'] != sha256(mixed_best):
        raise ValueError('Mixed comparison no longer points to its actual selected checkpoint')
    return dict(summary_sha256=config['mixed_reference_summary_sha256'],
                comparison_sha256=config['mixed_reference_comparison_sha256'],
                checkpoint_sha256=summary['checkpoint_sha256'])


def source_identity():
    files = shared.source_identity()
    for name in ('night_only_train.py', 'night_specialist.py', 'night_finalize.py',
                 'night_only_launch.sh'):
        path = 'reproduction/' + name
        files[path] = sha256(path)
    return files


def exposure_budget(config, num_train):
    updates = math.ceil(num_train / config['night_batch_size']) * config['epochs']
    return dict(num_training_nights=num_train, epochs=config['epochs'],
                optimizer_steps=updates, night_image_exposures=num_train * config['epochs'],
                day_query_image_exposures=0, total_student_image_exposures=num_train * config['epochs'],
                night_batch_size=config['night_batch_size'], replay_batch_size=0,
                reference_experiment='boq_b2_night_v1',
                comparison='Same initialization, frozen split, night exposures, update count, LR schedule; '
                           'day-query forward passes are removed, including their BatchNorm contribution')


class DiagnosticValidationLogger:
    """Keep the shared MSLS evaluator but label its scalars as diagnostics."""
    def __init__(self, logger):
        self.logger = logger

    def log(self, step, **values):
        self.logger.log(step, **{key.replace('val/msls_guard/', 'diagnostic/msls/'): value
                                 for key, value in values.items()})


def validation(model, config, epoch, step, logger):
    return shared.validation(model, config, epoch, step, DiagnosticValidationLogger(logger))


def train(config, all_records, lineage, logger, max_steps=None):
    out = Path(config['output_dir'])
    records = load_records(config, 'train')
    steps_per_epoch = math.ceil(len(records) / config['night_batch_size'])
    cache = Path(config['base_cache'])
    # Source-day descriptors are diagnostic only, never positive targets.
    day_teacher = np.load(cache / 'train_features.npy', mmap_mode='r')
    night_teacher = np.load(config['night_teacher_cache'], mmap_mode='r')
    centroids = torch.from_numpy(np.load(cache / 'centroids.npy', mmap_mode='c')).cuda()
    variances = torch.from_numpy(np.load(cache / 'variances.npy', mmap_mode='c')).cuda()
    model = EfficientViTBoQ(config, pretrained=False).cuda()
    optimizer = torch.optim.AdamW([
        {'params': model.backbone.parameters(), 'lr': config['lr_backbone'], 'name': 'backbone'},
        {'params': model.aggregator.parameters(), 'lr': config['lr_aggregator'], 'name': 'aggregator'},
    ], weight_decay=config['weight_decay'])
    ratio = config['min_lr_ratio']
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda ep: ratio + (1-ratio) * (1+math.cos(math.pi*ep/config['epochs']))/2)
    epoch0 = batch0 = step = 0
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        epoch0, batch0, step, best, baseline = shared.restore_state(
            state, model, optimizer, scheduler, config, lineage, steps_per_epoch)
        del state
        print(f'Resumed NIGHT-ONLY experiment epoch={epoch0} batch={batch0} step={step}', flush=True)
    else:
        initial = load_inference_checkpoint(config['init_checkpoint'])
        model.load_state_dict(initial['model'], strict=True)
        del initial
        baseline = validation(model, config, 0, 0, logger)
        atomic_json(out / 'baseline.json', baseline)
        score = float(baseline['svox_dev']['metrics_percent']['R@1'])
        best = dict(score=score, epoch=0, step=0, adapted=False,
            reason='Retain initialization until SVOX real-night development R@1 improves; no daytime veto')
        atomic_torch_save(out / 'best.pt', dict(model=model.state_dict(), config=config,
            epoch=0, step=0, metrics=baseline, provenance=lineage['base_provenance'], adaptation_lineage=lineage))
        shared.seed_all(config['seed'])
        atomic_torch_save(out / 'last.pt', shared.checkpoint_payload(model, optimizer, scheduler,
            0, 0, 0, best, baseline, config, lineage))
    model.train()
    if max_steps is not None and step >= max_steps:
        print(f'Bounded check already reached step {step}; no additional update started.', flush=True)
        return
    start = time.monotonic()
    accumulator, observations, samples = {}, 0, 0
    for epoch in range(epoch0, config['epochs']):
        dataset = NightOnlyQueries(records, all_records, config, epoch)
        batches = DataLoader(dataset, batch_size=config['night_batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] else None,
            generator=torch.Generator().manual_seed(config['seed']+epoch), drop_last=False)
        for batch_idx, (night, ni, source_idx, place) in enumerate(batches):
            if epoch == epoch0 and batch_idx < batch0:
                continue
            if shared.STOP:
                atomic_torch_save(out / 'last.pt', shared.checkpoint_payload(model, optimizer, scheduler,
                    epoch, batch_idx, step, best, baseline, config, lineage))
                raise SystemExit(130)
            optimizer.zero_grad(set_to_none=True)
            # Only night images pass through the student, including its BatchNorm.
            features = model(night.cuda(non_blocking=True))
            loss, details = night_only_objective(features,
                teacher_night=torch.from_numpy(np.array(night_teacher[ni.numpy()], copy=True)).cuda(),
                teacher_source_day=torch.from_numpy(np.array(day_teacher[source_idx.numpy()], copy=True)).cuda(),
                night_place_ids=place.cuda(), centroids=centroids, variances=variances,
                temperature=config['temperature'], gamma=config['gamma'], chunk_size=config['bank_chunk_size'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite night-only loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            observations += 1
            samples += len(night)
            for key, value in {'loss': loss.detach(), **details}.items():
                accumulator[key] = accumulator.get(key, 0.) + float(value)
            if step == 1 or step % config['log_every'] == 0:
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                values = {f'train/{k}': v/observations for k, v in accumulator.items()}
                values.update({'epoch': epoch+1, 'train/images_per_second': samples/elapsed,
                    'train/grad_norm': float(grad), 'lr/backbone': optimizer.param_groups[0]['lr'],
                    'lr/aggregator': optimizer.param_groups[1]['lr'],
                    'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
                logger.log(step, **values)
                print(f'epoch={epoch+1}/{config["epochs"]} step={step} '
                      f'loss={values["train/loss"]:.5f} night_images/s={samples/elapsed:.1f}', flush=True)
                accumulator, observations, samples = {}, 0, 0
                start = time.monotonic()
            if step % config['save_every'] == 0 or shared.STOP or (max_steps is not None and step >= max_steps):
                atomic_torch_save(out / 'last.pt', shared.checkpoint_payload(model, optimizer, scheduler,
                    epoch, batch_idx+1, step, best, baseline, config, lineage))
            if shared.STOP:
                raise SystemExit(130)
            if max_steps is not None and step >= max_steps:
                print('Reached bounded NIGHT-ONLY check; resume this same state to continue.', flush=True)
                return
        del batches, dataset
        # Finish validation before committing the epoch/scheduler boundary.
        atomic_torch_save(out / 'last.pt', shared.checkpoint_payload(model, optimizer, scheduler,
            epoch, steps_per_epoch, step, best, baseline, config, lineage))
        metrics = validation(model, config, epoch+1, step, logger)
        eligible, score, day_delta = select_night_best(metrics, baseline)
        weights = dict(model=model.state_dict(), config=config, epoch=epoch+1, step=step,
            metrics=metrics, provenance=lineage['base_provenance'], adaptation_lineage=lineage)
        atomic_torch_save(out / f'epoch_{epoch+1:02d}.pt', weights)
        if eligible and score > best['score']:
            best = dict(score=score, epoch=epoch+1, step=step, adapted=True, msls_delta_pp=day_delta,
                        selection_policy='svox_night_r1', daytime_veto=False)
            atomic_torch_save(out / 'best.pt', weights)
        atomic_json(out / f'epoch_{epoch+1:02d}_metrics.json', dict(epoch=epoch+1, step=step,
            eligible_for_selection=eligible, score=score, msls_delta_pp=day_delta,
            selection_policy='svox_night_r1', daytime_veto=False, best=best, evaluation=metrics))
        logger.log(step, **{'selection/best_svox_night_r1': best['score'],
                           'diagnostic/msls_delta_pp': day_delta})
        scheduler.step()
        atomic_torch_save(out / 'last.pt', shared.checkpoint_payload(model, optimizer, scheduler,
            epoch+1, 0, step, best, baseline, config, lineage))
        print(f'epoch {epoch+1} complete: night_dev_R1={score:.4f}, '
              f'MSLS diagnostic delta={day_delta:+.4f}pp (no veto), best={best}', flush=True)
        model.train()
        start = time.monotonic()
    atomic_json(out / 'completed.json', dict(status='completed', epochs=config['epochs'], steps=step,
        best=best, config=config, lineage=lineage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_b2_night_only_v1.json')
    parser.add_argument('--stage', choices=['prepare', 'train', 'all'], default='all')
    parser.add_argument('--max-steps', type=int, help='Bounded initial check, resumable as this same run')
    parser.add_argument('--skip-final-evaluation', action='store_true')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    config = json.loads(Path(args.config).read_text())
    validate_night_only_config(config)
    mixed_reference = validate_reference_inputs(config)
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.train.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This night-only output already has an active process')
    existing = out / 'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise RuntimeError('Output belongs to another immutable configuration')
    if (out / 'completed.json').exists() and args.stage in ('all', 'train'):
        finished = json.loads((out / 'completed.json').read_text())
        if finished.get('config') != config or not (out / 'last.pt').is_file() or not (out / 'best.pt').is_file():
            raise RuntimeError('Inconsistent completed night-only experiment')
        print('Night-only experiment already completed; no duplicate training started.', flush=True)
        if not args.skip_final_evaluation:
            shared.finalize_completed(config)
        return
    initial = load_inference_checkpoint(config['init_checkpoint'])
    base_identity, _ = shared.validate_base(config, initial)
    del initial
    if not existing.exists():
        atomic_json(existing, config)
    configure_cuda()
    torch.set_num_threads(config['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    signal.signal(signal.SIGINT, shared.request_stop)
    signal.signal(signal.SIGTERM, shared.request_stop)
    shared.seed_all(config['seed'])
    log_dir = Path(os.environ.get('TRAIN_TERMINAL_RUN_DIR', str(out / 'logs')))
    with TrainingLogger(run_dir=log_dir, config=config) as logger:
        snapshot, records, night_identity = shared.prepare(config, logger)
    if args.stage == 'prepare':
        return
    budget = exposure_budget(config, len(load_records(config, 'train')))
    if config.get('total_optimizer_steps') != budget['optimizer_steps']:
        raise ValueError('Frozen optimizer-update budget differs from actual night data/batch/epochs')
    lineage = dict(init_sha256=sha256(config['init_checkpoint']), base_provenance=base_identity,
        snapshot=snapshot, night_teacher_identity=night_identity,
        svox_dev_manifest_sha256=sha256(config['svox_dev_manifest']),
        training_source_sha256=source_identity(), exposure_budget=budget, mixed_reference=mixed_reference)
    if (out / 'lineage.json').exists():
        if json.loads((out / 'lineage.json').read_text()) != lineage:
            raise RuntimeError('Training code or immutable inputs changed; inspect lineage before resuming')
    else:
        atomic_json(out / 'lineage.json', lineage)
        snapshot_root = out.parent / 'source_snapshot'
        for name in lineage['training_source_sha256']:
            target = snapshot_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(name, target)
    atomic_json(out / 'budget.json', budget)
    atomic_json(out / 'trainroom.json', dict(directory=str(log_dir), total_steps=budget['optimizer_steps']))
    with TrainingLogger(run_dir=log_dir, config=config, total_steps=budget['optimizer_steps']) as logger:
        train(config, records, lineage, logger, args.max_steps)
    if (out / 'completed.json').exists() and not args.skip_final_evaluation:
        shared.finalize_completed(config)


if __name__ == '__main__':
    main()
