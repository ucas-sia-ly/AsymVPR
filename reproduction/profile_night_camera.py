"""Bounded CPU loader measurement; no model, optimizer or training updates."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from reproduction.monitoring import TrainingLogger
from reproduction.night_data import load_records
from reproduction.night_specialist import NightOnlyQueries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--workers', type=int, nargs='+', default=[4, 8])
    p.add_argument('--samples', type=int, default=256)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text())
    all_rows, train = load_records(config, 'all'), load_records(config, 'train')
    dataset = NightOnlyQueries(train, all_rows, config, epoch=0)
    subset = Subset(dataset, range(min(args.samples, len(dataset))))
    results = []
    with TrainingLogger(config=dict(stage='CPU camera input profiling; zero optimizer updates', **config)) as logger:
        for workers in args.workers:
            loader = DataLoader(subset, batch_size=config['night_batch_size'], shuffle=False,
                num_workers=workers, prefetch_factor=1 if workers else None,
                generator=torch.Generator().manual_seed(config['seed']))
            start = time.monotonic()
            digest, count, applied = hashlib.sha256(), 0, 0
            for images, ni, source, label, flag in loader:
                count += len(images)
                applied += int(flag.sum())
                digest.update(images.numpy().tobytes())
            elapsed = time.monotonic() - start
            row = dict(workers=workers, images=count, elapsed_seconds=elapsed,
                images_per_second=count/elapsed, applied_images=applied, tensor_sha256=digest.hexdigest(),
                includes='worker startup, original-size degradation, transform, IPC and tensor checksum')
            print(json.dumps(row), flush=True)
            logger.log(0, **{f'profile/workers_{workers}/images_per_second': count/elapsed,
                            f'profile/workers_{workers}/elapsed_seconds': elapsed})
            results.append(row)
    if len({r['tensor_sha256'] for r in results}) != 1:
        raise RuntimeError('Camera samples changed across loader worker counts')
    target = Path(args.output); target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(stage='CPU input profiling, not training', results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
