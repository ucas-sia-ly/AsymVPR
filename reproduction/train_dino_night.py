"""Fine-tune pretrained DINOv2-B + BoQ on GSV night queries in its fixed gallery space."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from reproduction.asym_model import GalleryBoQ
from reproduction.data import transform
from reproduction.evaluate import TRANSFORM, atomic_json, sha256
from reproduction.monitoring import TrainingLogger
from reproduction.night_data import load_records
from reproduction.night_degradation import as_image, degrade, statistics
from reproduction.night_eval import evaluate_msls_guard, evaluate_svox_dev
from reproduction.night_train import seed_all
from reproduction.run import atomic_torch_save, configure_cuda


STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def read_tensor(path, preprocess):
    with Image.open(path) as image:
        return preprocess(image.convert('RGB')).clamp_(0, 1)


class NightQueries(Dataset):
    def __init__(self, rows, config, epoch, cutoff):
        self.rows, self.seed, self.epoch, self.cutoff = rows, config['seed'], epoch, cutoff
        clean = transform(config['image_size']).transforms
        self.preprocess, self.normalize = T.Compose(clean[:-1]), clean[-1]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Preserve the exact existing transform on original and fallback branches.
        with Image.open(row['night_path']) as image:
            clean = self.preprocess(image.convert('RGB'))
        if self.cutoff is None:
            return self.normalize(clean), row['source_index'], torch.zeros(3, dtype=torch.bool)
        pixels = clean.clamp(0, 1)
        augmented, operations = degrade(pixels, row, self.seed, self.epoch, self.cutoff)
        return (self.normalize(clean if augmented is pixels else augmented), row['source_index'],
                torch.tensor(operations))


def edge_threshold(rows, config):
    cities = defaultdict(list)
    for row in rows:
        cities[row['city']].append(row)
    rng = random.Random(20261119)
    paths = [r['night_path'] for city in sorted(cities)
             for r in rng.sample(cities[city], min(24, len(cities[city])))]
    preprocess = T.Compose(transform(config['image_size']).transforms[:-1])

    def measure(path):
        return statistics(as_image(read_tensor(path, preprocess)))[1]

    with ThreadPoolExecutor(max_workers=config['cpu_threads']) as pool:
        cutoff = float(np.quantile(list(pool.map(measure, paths)), .25))
    return cutoff, len(paths)


def fixed_bank(config, snapshot):
    path = Path(config['base_cache'])
    provenance = json.loads((path / 'provenance.json').read_text())
    expected = dict(teacher_sha256=sha256(config['teacher_weights']),
                    image_size=config['image_size'], descriptor_dim=config['descriptor_dim'],
                    transform=TRANSFORM, precision='float32')
    if (any(provenance.get(k) != v for k,v in expected.items())
            or json.loads((path / 'bank_complete.json').read_text()) != provenance):
        raise ValueError('Fixed GSV gallery descriptors do not match the pretrained BoQ model')
    bank = np.load(path / 'train_features.npy', mmap_mode='r', allow_pickle=False)
    bank_identity = snapshot['base_cache_identity']['train_features.npy']
    if list(bank.shape) != bank_identity['shape'] or str(bank.dtype) != bank_identity['dtype']:
        raise ValueError('Fixed gallery descriptor shape or dtype changed')
    return bank, expected['teacher_sha256']


def validate(model, config, epoch, step, logger):
    return {name: fn(model, config, epoch=epoch, step=step, logger=logger)['metrics_percent']
            for name, fn in [('svox_night_dev', evaluate_svox_dev), ('msls_val', evaluate_msls_guard)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_dinob_gsv_night_degraded.json')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    overrides = json.loads(Path(args.config).read_text())
    base = json.loads(Path(overrides['base_config']).read_text())
    config = {**base, **overrides}
    config.pop('init_checkpoint')
    config.pop('student_name')
    snapshot = json.loads((Path(config['input_dir']) / 'summary.json').read_text())
    rows = load_records(config, 'train')
    if not rows or len({row['label'] for row in rows}) != len(rows):
        raise ValueError('Direct night contrastive training expects one night image per place')
    steps_per_epoch = math.ceil(len(rows)/config['batch_size'])
    total_steps = steps_per_epoch * config['epochs']
    torch.set_num_threads(config['cpu_threads'])
    bank, checkpoint_sha = fixed_bank(config, snapshot)
    use_degradation = config.get('use_degradation', True)
    if type(use_degradation) is not bool:
        raise ValueError('use_degradation must be a JSON boolean')
    cutoff, cutoff_count = edge_threshold(rows, config) if use_degradation else (None, 0)
    lineage = dict(snapshot_sha256=snapshot['manifest_sha256'], pretrained_sha256=checkpoint_sha,
                   base_cache_identity=snapshot['base_cache_identity'], edge_cutoff=cutoff,
                   edge_calibration_images=cutoff_count, source_sha256={
                       name: sha256(name) for name in ('reproduction/train_dino_night.py',
                       'reproduction/night_degradation.py', 'reproduction/camera_augmentation.py',
                       'reproduction/asym_model.py', 'src/backbones.py', 'src/boq.py')})
    transform_summary = (f'edge threshold {cutoff:.6f} from {cutoff_count} images'
                         if use_degradation else 'degradation disabled')
    print(f'{len(rows)} night images, {total_steps} updates, {transform_summary}', flush=True)
    if args.preflight:
        data = NightQueries(rows[:32], config, 0, cutoff)
        transformed = [data[i] for i in range(len(data))]
        assert all(image.shape == (3, config['image_size'], config['image_size'])
                   and torch.isfinite(image).all() for image, _, _ in transformed)
        if use_degradation:
            assert any(ops[:2].any() for _, _, ops in transformed)
        else:
            clean = transform(config['image_size'])
            for row, (tensor, _, ops) in zip(rows, transformed):
                with Image.open(row['night_path']) as image:
                    assert torch.equal(tensor, clean(image.convert('RGB')))
                assert not ops.any()
        print('CPU preflight passed', flush=True)
        return

    out = Path(config['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = dict(config=config, lineage=lineage)
        receipt = out / 'experiment.json'
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise RuntimeError('Output directory belongs to a different experiment')
        if (out / 'completed.json').is_file():
            print('Night experiment already completed', flush=True)
            return
        atomic_json(receipt, identity)
        configure_cuda()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision('highest')
        seed_all(config['seed'])
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, request_stop)
        model = GalleryBoQ(config).cuda()
        late = list(model.backbone.dino.blocks[-2:].parameters())
        head = list(model.aggregator.parameters())
        for parameter in late + head:
            parameter.requires_grad_(True)
        opt = torch.optim.AdamW([dict(params=late, lr=config['lr_backbone']),
                                 dict(params=head, lr=config['lr_aggregator'])],
                                weight_decay=config['weight_decay'])
        log_dir = os.environ.get('TRAIN_TERMINAL_RUN_DIR', str(out / 'logs'))
        with TrainingLogger(run_dir=log_dir, config=config, total_steps=total_steps) as logger:
            run(model, opt, bank, rows, config, identity, steps_per_epoch, out, logger)


def run(model, opt, bank, rows, config, identity, steps_per_epoch, out, logger):
    file = out / 'last.pt'
    epoch_start = next_batch = 0
    best_score = -1.
    if file.exists():
        state = torch.load(file, map_location='cpu', weights_only=True)
        if state['identity'] != identity:
            raise RuntimeError('Resume model or input lineage changed')
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        epoch_start, next_batch, best_score = state['epoch'], state['next_batch'], state['best_score']
        baseline = state['baseline']
        del state
    else:
        baseline = validate(model, config, 0, 0, logger)
        atomic_json(out / 'baseline.json', baseline)
    print(f'Pretrained validation: {baseline}', flush=True)

    def save(epoch, batch):
        atomic_torch_save(file, dict(model=model.state_dict(), optimizer=opt.state_dict(),
                          identity=identity, baseline=baseline, epoch=epoch, next_batch=batch,
                          best_score=best_score))

    for epoch in range(epoch_start, config['epochs']):
        model.train()
        data = NightQueries(rows, config, epoch, identity['lineage']['edge_cutoff'])
        order = np.random.default_rng(config['seed'] + epoch).permutation(len(rows))
        batches = DataLoader(data, batch_size=config['batch_size'],
            sampler=order[next_batch*config['batch_size']:].tolist(), num_workers=config['num_workers'],
            prefetch_factor=1, pin_memory=True)
        window_loss = torch.zeros((), device='cuda')
        window_flags = torch.zeros(4, dtype=torch.int64)
        window_batches = window_images = 0
        start = time.monotonic()
        for batch, (images, indices, flags) in enumerate(batches, next_batch + 1):
            opt.zero_grad(set_to_none=True)
            targets = torch.from_numpy(np.array(bank[indices.numpy()], copy=True)).cuda()
            queries = model(images.cuda(non_blocking=True))
            logits = queries @ targets.T / config['temperature']
            loss = F.cross_entropy(logits, torch.arange(len(indices), device='cuda'))
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(opt.param_groups[0]['params'] +
                                                   opt.param_groups[1]['params'], 1.,
                                                   error_if_nonfinite=True)
            opt.step()
            step = epoch*steps_per_epoch + batch
            window_loss += loss.detach()
            window_flags[:3] += flags.sum(dim=0)
            window_flags[3] += (flags[:,0] & flags[:,1]).sum()
            window_batches += 1
            window_images += len(indices)
            if step == 1 or step % config['log_every'] == 0 or batch == steps_per_epoch:
                value = float(window_loss/window_batches)
                sats, blurs, rejects, both = (window_flags/window_images).tolist()
                rate = window_images/(time.monotonic()-start)
                logger.log(step, **{'train/loss': value, 'train/gradient_norm': float(grad),
                    'train/saturation_fraction': sats, 'train/defocus_fraction': blurs,
                    'train/degradation_fraction': sats+blurs-both, 'train/both_fraction': both,
                    'train/fallback_fraction': rejects, 'train/images_per_second': rate,
                    'lr/backbone': opt.param_groups[0]['lr'], 'lr/boq': opt.param_groups[1]['lr']})
                print(f'epoch {epoch+1} step {step} loss={value:.5f} images/s={rate:.1f} '
                      f'degraded={sats+blurs-both:.2%}', flush=True)
                window_loss.zero_()
                window_flags.zero_()
                window_batches = window_images = 0
                start = time.monotonic()
            if step % config['save_every'] == 0 or batch == steps_per_epoch or STOP:
                save(epoch, batch)
            if STOP:
                raise SystemExit(130)
        del batches
        step = (epoch+1)*steps_per_epoch
        metrics = validate(model, config, epoch+1, step, logger)
        atomic_json(out / f'epoch_{epoch+1:02d}.json', metrics)
        score = metrics['svox_night_dev']['R@1']
        eligible = metrics['msls_val']['R@1'] >= baseline['msls_val']['R@1'] - config['max_msls_drop_pp']
        if eligible and score > max(best_score, baseline['svox_night_dev']['R@1']):
            best_score = score
            atomic_torch_save(out / 'best.pt',
                              dict(model=model.state_dict(), identity=identity, metrics=metrics, step=step))
        atomic_torch_save(out / f'epoch_{epoch+1:02d}.pt',
                          dict(model=model.state_dict(), identity=identity, metrics=metrics, step=step))
        save(epoch+1, 0)
        next_batch = 0
        print(f'Epoch {epoch+1} validation: {metrics}; MSLS guard={eligible}', flush=True)
    atomic_json(out / 'completed.json', dict(steps=steps_per_epoch*config['epochs'],
                improved_checkpoint=(out / 'best.pt').is_file(), pretrained_validation=baseline))


if __name__ == '__main__':
    main()
