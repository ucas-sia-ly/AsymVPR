"""Continue the compatible BoQ query model on a frozen generated-night snapshot."""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import time

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ
from reproduction.data import transform
from reproduction.evaluate import (TRANSFORM, atomic_json, cache_complete,
    extract_features, feature_identity, load_inference_checkpoint, sha256,
    validate_inference_checkpoint)
from reproduction.monitoring import TrainingLogger
from reproduction.night_data import build_snapshot, load_records
from reproduction.night_objective import night_objective
from reproduction.run import atomic_torch_save, configure_cuda

STOP = False


def request_stop(*_):
    global STOP
    STOP = True
    import reproduction.evaluate as extraction
    extraction.STOP = True
    print('Stop requested: saving the complete state at the next batch boundary.', flush=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_identity():
    files = ['reproduction/' + name for name in (
        'night_train.py', 'night_data.py', 'night_eval.py', 'night_objective.py',
        'asym_model.py', 'data.py', 'evaluate.py', 'benchmarks.py', 'path_resolver.py',
        'run.py', 'monitoring.py')]
    files += ['src/boq.py', 'src/backbones.py']
    return {name: sha256(name) for name in files}


def finalize_completed(config):
    # Models/optimizers from train() have left scope before the evaluation child.
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    from reproduction.night_finalize import finalize
    return finalize(config)


def validate_base(config, state):
    """Legacy cache is read-only; match its actual training lineage, never upgrade it."""
    cache = Path(config['base_cache'])
    identity = json.loads((cache / 'provenance.json').read_text())
    manifest = json.loads((cache / 'manifest.json').read_text())
    validate_inference_checkpoint(state, config, sha256(config['teacher_weights']))
    for key in ('manifest', 'teacher_sha256', 'image_size', 'descriptor_dim', 'transform', 'precision'):
        if identity.get(key) != state['provenance'].get(key):
            raise RuntimeError(f'Base cache differs from initialization checkpoint: {key}')
    if identity['manifest'] != manifest['signature']:
        raise RuntimeError('Base cache manifest signature mismatch')
    if json.loads((cache / 'bank_complete.json').read_text()) != identity:
        raise RuntimeError('Base geographical bank is incomplete or has another provenance')
    shapes = {'paths': (manifest['num_images'],), 'labels': (manifest['num_images'],),
              'offsets': (manifest['num_places'] + 1,),
              'train_features': (manifest['num_images'], config['descriptor_dim']),
              'centroids': (manifest['num_places'], config['descriptor_dim']),
              'variances': (manifest['num_places'], config['descriptor_dim'])}
    dtypes = {'paths': np.dtype('S256'), 'labels': np.dtype('int32'), 'offsets': np.dtype('int64')}
    for name, shape in shapes.items():
        arr = np.load(cache / f'{name}.npy', mmap_mode='r', allow_pickle=False)
        if arr.shape != shape or arr.dtype != dtypes.get(name, np.dtype('float32')):
            raise RuntimeError(f'Invalid base cache shape/dtype: {name}')
    progress = json.loads((cache / 'train_features.progress.json').read_text())
    if progress.get('rows') != manifest['num_images'] or progress.get('total') != manifest['num_images']:
        raise RuntimeError('Base gallery extraction is incomplete')
    return identity, manifest


class PairedReplay(Dataset):
    """One night pass plus uniformly sampled global day views, without val-place replay."""
    def __init__(self, records, all_records, config, epoch):
        self.config, self.records = config, records
        cache = Path(config['base_cache'])
        self.paths = np.load(cache / 'paths.npy', mmap_mode='r')
        self.labels = np.load(cache / 'labels.npy', mmap_mode='r')
        val_places = np.unique([r['label'] for r in all_records if r.get('split') == 'val'])
        # Also tolerate snapshot implementations that expose split via load_records only.
        if not len(val_places):
            val_places = np.unique([r['label'] for r in load_records(config, 'val')])
        eligible = np.flatnonzero(~np.isin(self.labels, val_places))
        if not len(eligible):
            raise ValueError('No day-replay images remain after excluding validation places')
        rng = np.random.default_rng(config['seed'] + epoch)
        self.order = rng.permutation(len(records))
        self.replay = rng.choice(eligible, size=len(records), replace=True)
        self.all_index = {r['night_path']: i for i, r in enumerate(all_records)}
        self.transform = transform(config['image_size'])

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        row = self.records[int(self.order[index])]
        replay_index = int(self.replay[index])
        name = self.paths[replay_index].decode()
        with Image.open(row['night_path']) as image:
            night = self.transform(image.convert('RGB'))
        with Image.open(Path(self.config['gsv_root']) / name) as image:
            day = self.transform(image.convert('RGB'))
        return night, day, self.all_index[row['night_path']], row['source_index'], row['label'], replay_index


class PreparationLogger:
    """Feature counts are metrics at optimizer step zero, not fictitious updates."""
    def __init__(self, logger):
        self.logger = logger

    def log(self, _step, **values):
        self.logger.log(0, **values)


def prepare(config, logger):
    from reproduction.night_eval import prepare_eval_caches
    snapshot = build_snapshot(config)
    records = load_records(config, 'all')
    paths = [Path(r['night_path']) for r in records]
    teacher_sha = sha256(config['teacher_weights'])
    identity = feature_identity(paths, config, teacher_sha, 'night_teacher')
    target = Path(config['night_teacher_cache'])
    teacher = None
    if not cache_complete(target, identity):
        teacher = GalleryBoQ(config).cuda().eval()
        extract_features(teacher, paths, target, identity, config, 'cuda',
            config['extract_batch_size'], config['num_workers'], PreparationLogger(logger), 'prepare/night_teacher')
    if STOP:
        raise SystemExit(130)
    prepare_eval_caches(config, PreparationLogger(logger), teacher=teacher, device='cuda')
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    return snapshot, records, identity


def validation(model, config, epoch, step, logger):
    from reproduction.night_eval import evaluate_synthetic, evaluate_msls_guard, evaluate_svox_dev
    result = {}
    for name, fn in [('synthetic', evaluate_synthetic), ('msls', evaluate_msls_guard), ('svox_dev', evaluate_svox_dev)]:
        if STOP:
            raise SystemExit(130)
        result[name] = fn(model, config, device='cuda', epoch=epoch, step=step, logger=logger)
        print(f'validation/{name}: {result[name]["metrics_percent"]}', flush=True)
    return result


def checkpoint_payload(model, optimizer, scheduler, epoch, next_batch, step, best, baseline, config, lineage):
    return dict(schema=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), epoch=epoch, next_batch=next_batch, step=step,
        best=best, baseline=baseline, config=config, lineage=lineage,
        random_state=random.getstate(), numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all())


def restore_state(state, model, optimizer, scheduler, config, lineage, steps_per_epoch):
    if state['config'] != config or state['lineage'] != lineage:
        raise RuntimeError('Night resume config or immutable input lineage differs')
    epoch, batch, step = state['epoch'], state['next_batch'], state['step']
    if (any(type(v) is not int for v in (epoch, batch, step))
            or not 0 <= epoch <= config['epochs'] or not 0 <= batch <= steps_per_epoch
            or (epoch == config['epochs'] and batch != 0)
            or step != epoch * steps_per_epoch + batch
            or state['scheduler']['last_epoch'] != epoch):
        raise RuntimeError('Invalid night resume counters/scheduler')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    random.setstate(state['random_state'])
    np.random.set_state(state['numpy_state'])
    torch.set_rng_state(state['torch_state'])
    torch.cuda.set_rng_state_all(state['cuda_state'])
    return epoch, batch, step, state['best'], state['baseline']


def eligible_selection(metrics, baseline, max_msls_drop_pp):
    day = metrics['msls']['metrics_percent']['R@1']
    floor = baseline['msls']['metrics_percent']['R@1'] - max_msls_drop_pp
    score = metrics['svox_dev']['metrics_percent']['R@1']
    return bool(day >= floor), float(score), float(day - baseline['msls']['metrics_percent']['R@1'])


def train(config, all_records, lineage, logger, max_steps=None):
    out = Path(config['output_dir'])
    records = load_records(config, 'train')
    steps_per_epoch = math.ceil(len(records) / config['night_batch_size'])
    cache = Path(config['base_cache'])
    # Keep descriptors on disk; only fixed place statistics reside on the GPU.
    day_teacher = np.load(cache / 'train_features.npy', mmap_mode='r')
    night_teacher = np.load(config['night_teacher_cache'], mmap_mode='r')
    labels = np.load(cache / 'labels.npy', mmap_mode='r')
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
    best = baseline = None
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        epoch0, batch0, step, best, baseline = restore_state(state, model, optimizer, scheduler,
            config, lineage, steps_per_epoch)
        del state
        print(f'Resumed night experiment epoch={epoch0} batch={batch0} step={step}', flush=True)
    else:
        initial = load_inference_checkpoint(config['init_checkpoint'])
        model.load_state_dict(initial['model'], strict=True)
        del initial
        baseline = validation(model, config, 0, 0, logger)
        atomic_json(out / 'baseline.json', baseline)
        score = float(baseline['svox_dev']['metrics_percent']['R@1'])
        best = dict(score=score, epoch=0, step=0, adapted=False,
                    reason='Initial checkpoint retained until an adaptation improves real-night development R@1 within the day guard')
        atomic_torch_save(out / 'best.pt', dict(model=model.state_dict(), config=config,
            epoch=0, step=0, metrics=baseline, provenance=lineage['base_provenance'], adaptation_lineage=lineage))
        # Preparation/validation may consume RNG; train initialization is independent of cache reuse.
        seed_all(config['seed'])
        atomic_torch_save(out / 'last.pt', checkpoint_payload(model, optimizer, scheduler,
            0, 0, 0, best, baseline, config, lineage))
    model.train()
    if max_steps and step >= max_steps:
        print(f'Bounded check already reached step {step}; no additional update started.', flush=True)
        return
    start = time.monotonic()
    accumulator, observations, samples = {}, 0, 0
    for epoch in range(epoch0, config['epochs']):
        dataset = PairedReplay(records, all_records, config, epoch)
        batches = DataLoader(dataset, batch_size=config['night_batch_size'], shuffle=False,
            num_workers=config['num_workers'], pin_memory=True,
            prefetch_factor=config['prefetch_factor'] if config['num_workers'] else None,
            generator=torch.Generator().manual_seed(config['seed']+epoch), drop_last=False)
        for batch_idx, (night, day, ni, source_idx, place, replay_idx) in enumerate(batches):
            if epoch == epoch0 and batch_idx < batch0:
                continue
            if STOP:
                atomic_torch_save(out / 'last.pt', checkpoint_payload(model, optimizer, scheduler,
                    epoch, batch_idx, step, best, baseline, config, lineage))
                raise SystemExit(130)
            optimizer.zero_grad(set_to_none=True)
            count = len(night)
            images = torch.cat((night, day), 0).cuda(non_blocking=True)
            features = model(images)
            kwargs = dict(
                teacher_night=torch.from_numpy(np.array(night_teacher[ni.numpy()], copy=True)).cuda(),
                teacher_source_day=torch.from_numpy(np.array(day_teacher[source_idx.numpy()], copy=True)).cuda(),
                teacher_replay_day=torch.from_numpy(np.array(day_teacher[replay_idx.numpy()], copy=True)).cuda(),
                night_place_ids=place.cuda(),
                replay_place_ids=torch.from_numpy(np.asarray(labels[replay_idx.numpy()], dtype=np.int64)).cuda(),
                centroids=centroids, variances=variances,
                temperature=config['temperature'], gamma=config['gamma'], chunk_size=config['bank_chunk_size'],
                gate_cos_min=config['gate_cos_min'], gate_cos_span=config['gate_cos_span'],
                max_night_weight=config['max_night_weight'])
            loss, details = night_objective(features[:count], features[count:], **kwargs)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite night adaptation loss at step {step}')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            observations += 1
            samples += count * 2
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
                print(f'epoch={epoch+1}/{config["epochs"]} step={step} loss={values["train/loss"]:.5f} '
                      f'night={values["train/night_asym"]:.5f} day={values["train/replay_asym"]:.5f} '
                      f'images/s={values["train/images_per_second"]:.1f}', flush=True)
                accumulator, observations, samples = {}, 0, 0
                start = time.monotonic()
            if step % config['save_every'] == 0 or STOP or (max_steps and step >= max_steps):
                atomic_torch_save(out / 'last.pt', checkpoint_payload(model, optimizer, scheduler,
                    epoch, batch_idx+1, step, best, baseline, config, lineage))
            if STOP:
                raise SystemExit(130)
            if max_steps and step >= max_steps:
                print('Reached bounded training check; resume this same state to continue.', flush=True)
                return
        del batches, dataset
        # A validation interruption resumes at the completed batch boundary,
        # reruns validation and commits the scheduler/epoch only afterwards.
        atomic_torch_save(out / 'last.pt', checkpoint_payload(model, optimizer, scheduler,
            epoch, steps_per_epoch, step, best, baseline, config, lineage))
        metrics = validation(model, config, epoch+1, step, logger)
        eligible, score, day_delta = eligible_selection(metrics, baseline, config['max_msls_drop_pp'])
        weights = dict(model=model.state_dict(), config=config, epoch=epoch+1, step=step,
                       metrics=metrics, provenance=lineage['base_provenance'], adaptation_lineage=lineage)
        atomic_torch_save(out / f'epoch_{epoch+1:02d}.pt', weights)
        if eligible and score > best['score']:
            best = dict(score=score, epoch=epoch+1, step=step, adapted=True, msls_delta_pp=day_delta)
            atomic_torch_save(out / 'best.pt', weights)
        atomic_json(out / f'epoch_{epoch+1:02d}_metrics.json', dict(epoch=epoch+1, step=step,
                    eligible_for_selection=eligible, score=score, msls_delta_pp=day_delta, best=best, evaluation=metrics))
        scheduler.step()
        atomic_torch_save(out / 'last.pt', checkpoint_payload(model, optimizer, scheduler,
            epoch+1, 0, step, best, baseline, config, lineage))
        print(f'epoch {epoch+1} complete: night_dev_R1={score:.4f}, MSLS_delta={day_delta:+.4f}pp, best={best}', flush=True)
        model.train()
        start = time.monotonic()
    atomic_json(out / 'completed.json', dict(status='completed', epochs=config['epochs'], steps=step,
                best=best, config=config, lineage=lineage))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_b2_night_v1.json')
    parser.add_argument('--stage', choices=['all', 'snapshot', 'prepare', 'train'], default='all')
    parser.add_argument('--max-steps', type=int, help='Bounded, explicitly labelled training check; same run can resume')
    parser.add_argument('--skip-final-evaluation', action='store_true', help='Only train; default runs fixed final tests after completion')
    args = parser.parse_args()
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max-steps must be positive')
    config = json.loads(Path(args.config).read_text())
    if config['precision'] != 'float32' or config['night_batch_size'] != config['replay_batch_size']:
        raise ValueError('This experiment uses FP32 and equally sized night/day replay batches')
    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.train.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('This night adaptation output already has an active process')
    existing = out / 'config.json'
    if existing.exists() and json.loads(existing.read_text()) != config:
        raise RuntimeError('Output belongs to another immutable configuration')
    if (out / 'completed.json').exists() and args.stage in ('all', 'train'):
        finished = json.loads((out / 'completed.json').read_text())
        if finished.get('config') != config or not (out / 'last.pt').is_file() or not (out / 'best.pt').is_file():
            raise RuntimeError('Inconsistent completed night experiment')
        print('Night experiment already completed; no duplicate training started.', flush=True)
        if not args.skip_final_evaluation:
            finalize_completed(config)
        return
    initial = load_inference_checkpoint(config['init_checkpoint'])
    base_identity, _ = validate_base(config, initial)
    del initial
    if not existing.exists():
        atomic_json(existing, config)
    if args.stage == 'snapshot':
        print(json.dumps(build_snapshot(config), indent=2, ensure_ascii=False), flush=True)
        return
    configure_cuda()
    torch.set_num_threads(config['cpu_threads'])
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    seed_all(config['seed'])
    log_dir = Path(os.environ.get('TRAIN_TERMINAL_RUN_DIR', str(out)))
    with TrainingLogger(run_dir=log_dir, config=config) as logger:
        snapshot, records, night_identity = prepare(config, logger)
    if args.stage == 'prepare':
        return
    lineage = dict(init_sha256=sha256(config['init_checkpoint']), base_provenance=base_identity,
                   snapshot=snapshot, night_teacher_identity=night_identity,
                   svox_dev_manifest_sha256=sha256(config['svox_dev_manifest']),
                   training_source_sha256=source_identity())
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
    total_steps = math.ceil(len(load_records(config, 'train')) / config['night_batch_size']) * config['epochs']
    atomic_json(out / 'trainroom.json', dict(directory=str(log_dir), total_steps=total_steps))
    with TrainingLogger(run_dir=log_dir, config=config, total_steps=total_steps) as logger:
        train(config, records, lineage, logger, args.max_steps)
    if (out / 'completed.json').exists() and not args.skip_final_evaluation:
        finalize_completed(config)


if __name__ == '__main__':
    main()
