"""Usage: python -m reproduction.run --config configs/boq_b2.json"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import tempfile
import time

import numpy as np
import torch

from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ, asym_loss
from reproduction.data import Images, atomic_json, loader, manifest, msls, place_statistics
from reproduction.monitoring import TrainingLogger

STOP = False
MSLS_FILES = ('msls_val_dbImages.npy', 'msls_val_qImages.npy',
              'msls_val_qIdx.npy', 'msls_val_pIdx.npy')


def request_stop(*_):
    global STOP
    STOP = True
    print('Stop requested; saving at the next batch boundary.', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


def provenance(c, meta):
    # The ordered evaluation lists are part of the feature cache identity: a
    # same-length replacement must never reuse references in their old order.
    return dict(schema_version=2, manifest=meta['signature'],
                teacher_sha256=sha256(c['teacher_weights']),
                msls_sha256={name: sha256(Path(c['msls_root']) / name) for name in MSLS_FILES},
                image_size=c['image_size'], descriptor_dim=c['descriptor_dim'],
                transform='ToTensor/Resize-bicubic-antialias/ImageNet-normalize', precision=c['precision'])


def checked_array(path, shape, dtype):
    path = Path(path)
    try:
        array = np.load(path, mmap_mode='r', allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError(f'Invalid or missing cache array: {path}') from error
    if array.shape != tuple(shape) or array.dtype != np.dtype(dtype):
        raise RuntimeError(f'Cache shape/dtype mismatch: {path}; '
                           f'expected {tuple(shape)}/{np.dtype(dtype)}, got {array.shape}/{array.dtype}')
    return array


def extraction_rows(target, shape):
    """Check both the feature file and its durable progress before reusing it."""
    target = Path(target)
    status = target.with_suffix('.progress.json')
    progress = json.loads(status.read_text()) if status.exists() else {'rows': 0, 'total': shape[0]}
    rows = progress.get('rows')
    if (type(rows) is not int or not 0 <= rows <= shape[0]
            or progress.get('total') != shape[0]):
        raise RuntimeError(f'Invalid extraction progress: {status}')
    if target.exists():
        checked_array(target, shape, 'float32')
    elif rows:
        raise RuntimeError(f'Extraction progress exists without its feature file: {target}')
    return rows


def validate_manifest_arrays(c, meta):
    cache = Path(c['cache_dir'])
    count, places = meta['num_images'], meta['num_places']
    checked_array(cache / 'paths.npy', (count,), 'S256')
    labels = checked_array(cache / 'labels.npy', (count,), 'int32')
    offsets = checked_array(cache / 'offsets.npy', (places + 1,), 'int64')
    if (count <= 0 or places < 2 or offsets[0] != 0 or offsets[-1] != count
            or np.any(np.diff(offsets) <= 0) or labels.min() < 0 or labels.max() >= places):
        raise RuntimeError('Invalid training manifest labels/offsets')


def validate_cache(c, meta, val_data=None, identity=None):
    """Read-only preflight shared by prepare and the train-only entry point."""
    cache = Path(c['cache_dir'])
    identity = provenance(c, meta) if identity is None else identity
    provenance_path = cache / 'provenance.json'
    if not provenance_path.exists() or json.loads(provenance_path.read_text()) != identity:
        raise RuntimeError('Gallery cache provenance mismatch; use a new cache')
    validate_manifest_arrays(c, meta)
    val_data = msls(c) if val_data is None else val_data
    for name, count in [('train_features', meta['num_images']), ('msls_features', len(val_data[0]))]:
        target = cache / f'{name}.npy'
        if extraction_rows(target, (count, c['descriptor_dim'])) != count or not target.exists():
            raise RuntimeError(f'Gallery extraction is not complete: {name}')
    bank_path = cache / 'bank_complete.json'
    if not bank_path.exists() or json.loads(bank_path.read_text()) != identity:
        raise RuntimeError('Geographical bank provenance mismatch or preparation incomplete')
    for name in ('centroids', 'variances'):
        checked_array(cache / f'{name}.npy', (meta['num_places'], c['descriptor_dim']), 'float32')
    return val_data


def guard_output(c):
    """Reject another experiment before writing any configuration or logs."""
    out = Path(c['output_dir'])
    config_path = out / 'config.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != c:
            raise RuntimeError('Output directory belongs to a different configuration; choose a new output_dir')
    elif any(out.glob('*.pt')) or (out / 'completed.json').exists():
        raise RuntimeError('Existing experiment has no config.json; choose a new output_dir')


def training_complete(c):
    guard_output(c)
    out = Path(c['output_dir'])
    marker = out / 'completed.json'
    if not marker.exists():
        return False
    completed = json.loads(marker.read_text())
    best = completed.get('best_msls_r1')
    if (completed.get('config') != c or completed.get('epochs') != c['epochs']
            or type(completed.get('steps')) is not int or completed['steps'] <= 0
            or not isinstance(best, (int, float)) or not math.isfinite(best)
            or not 0 <= best <= 100
            or not completed.get('provenance')
            or any(not (out / name).is_file() for name in
                   ('last.pt', 'best.pt', f'epoch_{c["epochs"]:02d}.pt'))):
        raise RuntimeError('Invalid completed training record; inspect the output directory before resuming')
    return True


def configure_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError('Training/preparation requires a CUDA GPU; no CUDA device is available')
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info(0)
    reserve = min(4 * 1024**3, total * .1)
    budget = min(total - reserve, free - reserve)
    if budget <= 0:
        raise RuntimeError('Insufficient free CUDA memory after reserving desktop/validation headroom')
    torch.cuda.set_per_process_memory_fraction(min(budget / total, 1.), device=0)
    print(f'CUDA allocator limit: {budget / 1024**3:.1f} GiB; '
          f'free/total {free / 1024**3:.1f}/{total / 1024**3:.1f} GiB', flush=True)


def extract(model, dataset, target, c, logger, phase):
    target = Path(target)
    status_path = target.with_suffix('.progress.json')
    n = len(dataset.paths)
    rows = extraction_rows(target, (n, c['descriptor_dim']))
    if rows == n and target.exists():
        return
    features = np.lib.format.open_memmap(target, mode='r+' if target.exists() else 'w+',
        dtype='float32', shape=(n, c['descriptor_dim']))
    dataset.start = rows
    batches = loader(dataset, c['extract_batch_size'], c['num_workers'])
    then, initial = time.monotonic(), rows
    model.eval()
    for i, (images, indexes) in enumerate(batches):
        with torch.inference_mode():
            embeddings = model(images.cuda(non_blocking=True))
        if not torch.isfinite(embeddings).all():
            raise FloatingPointError(f'Nonfinite gallery embedding in {phase}')
        features[indexes.numpy()] = embeddings.cpu().numpy()
        done = int(indexes[-1]) + 1
        if i % 20 == 0 or done == n or STOP:
            features.flush()
            atomic_json(status_path, {'rows': done, 'total': n})
            rate = (done - initial) / (time.monotonic() - then)
            logger.log(done, **{f'{phase}/images': done, f'{phase}/images_per_second': rate})
            print(f'{phase}: {done}/{n}, {rate:.1f} images/s, ETA {(n-done)/max(rate,1)/60:.1f} min', flush=True)
        if STOP:
            raise SystemExit(130)
    dataset.start = 0
    del batches, features


def prepare(c, logger, meta):
    cache = Path(c['cache_dir'])
    identity = provenance(c, meta)
    provenance_path = cache / 'provenance.json'
    if provenance_path.exists() and json.loads(provenance_path.read_text()) != identity:
        raise RuntimeError('Gallery cache provenance mismatch; use a new cache')
    if not provenance_path.exists() and any((cache / name).exists() for name in
            ('train_features.npy', 'msls_features.npy', 'centroids.npy', 'variances.npy', 'bank_complete.json')):
        raise RuntimeError('Existing gallery artifacts have no provenance; use a new cache')
    bank_path = cache / 'bank_complete.json'
    if bank_path.exists() and json.loads(bank_path.read_text()) != identity:
        raise RuntimeError('Geographical bank provenance mismatch; use a new cache')
    atomic_json(provenance_path, identity)
    validate_manifest_arrays(c, meta)
    train_paths = np.load(cache / 'paths.npy', mmap_mode='r')
    val_paths, nrefs, positives = msls(c)
    teacher = None
    train_rows = extraction_rows(cache / 'train_features.npy', (len(train_paths), c['descriptor_dim']))
    val_rows = extraction_rows(cache / 'msls_features.npy', (len(val_paths), c['descriptor_dim']))
    if train_rows != len(train_paths) or val_rows != len(val_paths):
        teacher = GalleryBoQ(c).cuda().eval()
        print('Frozen gallery loaded strictly: DINOv2-B + BoQ, 12288-D, FP32', flush=True)
        extract(teacher, Images(c['gsv_root'], train_paths, c['image_size']),
                cache / 'train_features.npy', c, logger, 'gallery_gsv')
        extract(teacher, Images(c['msls_root'], val_paths, c['image_size']),
                cache / 'msls_features.npy', c, logger, 'gallery_msls')
        del teacher
        gc.collect(); torch.cuda.empty_cache()
    if not (cache / 'gallery_msls_metrics.json').exists():
        saved = np.load(cache / 'msls_features.npy', mmap_mode='r')
        refs_gpu = torch.from_numpy(saved[:nrefs].copy()).cuda()
        predicted = []
        with torch.inference_mode():
            for start in range(nrefs, len(saved), 128):
                q_gpu = torch.from_numpy(saved[start:start+128].copy()).cuda()
                predicted.extend((q_gpu @ refs_gpu.T).topk(10, dim=1).indices.cpu().numpy())
        recalls = {f'msls-val/R@{k}': float(100 * np.mean([
            np.isin(pred[:k], gt).any() for pred, gt in zip(predicted, positives)
        ])) for k in (1, 5, 10)}
        atomic_json(cache / 'gallery_msls_metrics.json', dict(protocol='symmetric frozen gallery baseline', **recalls))
        print(f'Frozen gallery MSLS baseline: {recalls}', flush=True)
        del saved, refs_gpu, q_gpu
        gc.collect(); torch.cuda.empty_cache()
    if not (cache / 'bank_complete.json').exists():
        print('Computing all geographic centroids and diagonal covariance matrices...', flush=True)
        place_statistics(np.load(cache / 'train_features.npy', mmap_mode='r'),
                         np.load(cache / 'offsets.npy'), cache)
        atomic_json(cache / 'bank_complete.json', identity)
    return validate_cache(c, meta, (val_paths, nrefs, positives), identity)


@torch.inference_mode()
def evaluate(model, c, paths, nrefs, positives):
    model.eval()
    ds = Images(c['msls_root'], paths[nrefs:], c['image_size'])
    query = torch.cat([model(x.cuda(non_blocking=True)).cpu()
        for x, _ in loader(ds, c['batch_size'], c['num_workers'])]).cuda()
    reference = torch.from_numpy(np.load(Path(c['cache_dir']) / 'msls_features.npy',
                                         mmap_mode='r')[:nrefs].copy()).cuda()
    predictions = torch.cat([(x @ reference.T).topk(10, dim=1).indices.cpu()
                              for x in query.split(128)]).numpy()
    recalls = {f'msls-val/R@{k}': 100 * np.mean([
        np.isin(pred[:k], gt).any() for pred, gt in zip(predictions, positives)
    ]) for k in (1, 5, 10)}
    model.train()
    return recalls


def atomic_torch_save(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            torch.save(value, file)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_checkpoint(path, model, optimizer, scheduler, epoch, next_batch, step, best, c, identity):
    value = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
        scheduler=scheduler.state_dict(), epoch=epoch, next_batch=next_batch,
        step=step, best=best, config=c, provenance=identity,
        random_state=random.getstate(), numpy_state=np.random.get_state(),
        torch_state=torch.get_rng_state(), cuda_state=torch.cuda.get_rng_state_all())
    atomic_torch_save(path, value)


def commit_epoch(out, model, optimizer, scheduler, epoch, step, best, recalls, c, identity):
    """Commit last only after all epoch artifacts, including best, are durable."""
    out = Path(out)
    improved = recalls['msls-val/R@1'] > best
    next_best = max(best, recalls['msls-val/R@1'])
    weights = dict(model=model.state_dict(), config=c, epoch=epoch, step=step,
                   metrics=recalls, provenance=identity)
    atomic_torch_save(out / f'epoch_{epoch:02d}.pt', weights)
    if improved:
        atomic_torch_save(out / 'best.pt', weights)
    atomic_json(out / f'epoch_{epoch:02d}_metrics.json', dict(epoch=epoch, step=step, **recalls))
    save_checkpoint(out / 'last.pt', model, optimizer, scheduler, epoch, 0, step, next_best, c, identity)
    return next_best


def validate_resume_state(state, c, identity, meta):
    if state['config'] != c or state['provenance'] != identity:
        raise RuntimeError('Resume configuration/provenance mismatch')
    epoch, batch, step = (state[k] for k in ('epoch', 'next_batch', 'step'))
    batches = math.ceil(meta['num_images'] / c['batch_size'])
    if (any(type(value) is not int for value in (epoch, batch, step))
            or not 0 <= epoch <= c['epochs'] or not 0 <= batch <= batches
            or (epoch == c['epochs'] and batch != 0)
            or step != epoch * batches + batch
            or state['scheduler'].get('last_epoch') != epoch):
        raise RuntimeError('Invalid resume epoch/batch/step/scheduler state')


def train(c, meta, logger, val_data, smoke_steps=0):
    cache, out = Path(c['cache_dir']), Path(c['output_dir'])
    if training_complete(c):
        print('Training already completed; no optimization was started.', flush=True)
        return
    validate_cache(c, meta, val_data)
    out.mkdir(parents=True, exist_ok=True)
    identity = json.loads((cache / 'provenance.json').read_text())
    means = torch.from_numpy(np.load(cache / 'centroids.npy', mmap_mode='c')).cuda()
    variances = torch.from_numpy(np.load(cache / 'variances.npy', mmap_mode='c')).cuda()
    features = np.load(cache / 'train_features.npy', mmap_mode='r')
    labels = np.load(cache / 'labels.npy', mmap_mode='r')
    paths = np.load(cache / 'paths.npy', mmap_mode='r')
    model = EfficientViTBoQ(c, pretrained=not (out / 'last.pt').exists()).cuda()
    print(f'Query model: {sum(p.numel() for p in model.parameters()):,} trainable parameters', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
        T_max=c['epochs'], eta_min=c['min_lr'])
    epoch0 = batch0 = step = 0
    best = -1.
    if (out / 'last.pt').exists():
        state = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        validate_resume_state(state, c, identity, meta)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        epoch0, batch0, step, best = (state[k] for k in ('epoch', 'next_batch', 'step', 'best'))
        random.setstate(state['random_state']); np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state']); torch.cuda.set_rng_state_all(state['cuda_state'])
        del state
        print(f'Resumed complete training state: epoch {epoch0}, batch {batch0}, step {step}', flush=True)
    dataset = Images(c['gsv_root'], paths, c['image_size'])
    since = time.monotonic()
    samples_since, loss_since, steps_since = 0, 0., 0
    for epoch in range(epoch0, c['epochs']):
        if STOP:
            raise SystemExit(130)
        model.train()
        # Dedicated per-epoch generator reconstructs exactly the same shuffled order.
        generator = torch.Generator().manual_seed(c['seed'] + epoch)
        batches = loader(dataset, c['batch_size'], c['num_workers'], True, generator)
        for batch_idx, (images, indexes) in enumerate(batches):
            if epoch == epoch0 and batch_idx < batch0:
                continue
            if STOP:
                save_checkpoint(out / 'last.pt', model, optimizer, scheduler,
                    epoch, batch_idx, step, best, c, identity)
                raise SystemExit(130)
            optimizer.zero_grad(set_to_none=True)
            idx = indexes.numpy()
            target = torch.from_numpy(features[idx].copy()).cuda()
            place_ids = torch.from_numpy(labels[idx].astype('int64')).cuda()
            q = model(images.cuda(non_blocking=True))
            loss = asym_loss(q, target, place_ids, means, variances,
                c['temperature'], c['gamma'], c['bank_chunk_size'])
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at step {step}')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            step += 1
            samples_since += len(images); loss_since += float(loss.detach()); steps_since += 1
            if step % c['log_every'] == 0 or step == 1:
                elapsed = time.monotonic() - since
                metrics = {'train/loss': loss_since / steps_since, 'train/grad_norm': float(grad_norm),
                    'learning_rate': optimizer.param_groups[0]['lr'], 'epoch': epoch + 1,
                    'train/images_per_second': samples_since / elapsed,
                    'gpu/peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
                logger.log(step, **metrics)
                print(f'epoch={epoch+1}/{c["epochs"]} step={step} loss={metrics["train/loss"]:.5f} '
                      f'images/s={metrics["train/images_per_second"]:.1f}', flush=True)
                since, samples_since, loss_since, steps_since = time.monotonic(), 0, 0., 0
            if step % c['save_every'] == 0 or STOP or (smoke_steps and step >= smoke_steps):
                save_checkpoint(out / 'last.pt', model, optimizer, scheduler,
                    epoch, batch_idx + 1, step, best, c, identity)
            if STOP:
                raise SystemExit(130)
            if smoke_steps and step >= smoke_steps:
                print('Bounded smoke training finished; production training is a separate experiment.', flush=True)
                return
        recalls = evaluate(model, c, *val_data)
        logger.log(step, **recalls)
        print(f'epoch {epoch+1}: {recalls}', flush=True)
        scheduler.step()
        best = commit_epoch(out, model, optimizer, scheduler, epoch+1, step, best, recalls, c, identity)
    atomic_json(out / 'completed.json', dict(epochs=c['epochs'], steps=step,
                best_msls_r1=float(best), config=c, provenance=identity))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/boq_b2.json')
    parser.add_argument('--stage', choices=['all', 'prepare', 'train', 'manifest'], default='all')
    args = parser.parse_args()
    c = json.loads(Path(args.config).read_text())
    guard_output(c)
    if args.stage in ('all', 'train') and training_complete(c):
        print('Training already completed; use reproduction.evaluate to evaluate the saved checkpoint.', flush=True)
        return
    meta = manifest(c['gsv_root'], c['cache_dir'])
    print(f'Dataset verified: {meta["num_images"]} images, {meta["num_places"]} places, {len(meta["cities"])} cities', flush=True)
    if args.stage == 'manifest':
        return
    if c['precision'] != 'float32':
        raise ValueError('This training implementation supports precision=float32 only')
    if args.stage == 'train':
        val_data = validate_cache(c, meta)
    configure_cuda()
    out = Path(c['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if not (out / 'config.json').exists():
        atomic_json(out / 'config.json', c)
    total_steps = math.ceil(meta['num_images'] / c['batch_size']) * c['epochs']
    logger_dir = Path(os.environ.get('TRAIN_TERMINAL_RUN_DIR', c['output_dir']))
    atomic_json(Path(c['output_dir']) / 'trainroom.json', dict(directory=str(logger_dir), total_steps=total_steps))
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    signal.signal(signal.SIGINT, request_stop); signal.signal(signal.SIGTERM, request_stop)
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
    if args.stage in ('all', 'prepare'):
        with TrainingLogger(run_dir=logger_dir / 'prepare', config=c) as prep_log:
            val_data = prepare(c, prep_log, meta)
    if args.stage in ('all', 'train'):
        if STOP:
            raise SystemExit(130)
        # Reset seeds so student initialization does not depend on cache reuse.
        random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.cuda.manual_seed_all(c['seed'])
        with TrainingLogger(run_dir=logger_dir, config=c, total_steps=total_steps) as log:
            train(c, meta, log, val_data)


if __name__ == '__main__':
    main()
