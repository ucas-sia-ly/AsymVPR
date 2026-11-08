"""Additional night protocols frozen from audited public metadata."""
import json
from pathlib import Path

import numpy as np

from reproduction.benchmarks import Benchmark, checked_positives, json_sha256, sha256


def load_protocol(path):
    path = Path(path)
    record = json.loads(path.read_text())
    body = {k: v for k, v in record.items() if k != 'record_sha256'}
    if record.get('record_sha256') != json_sha256(body):
        raise ValueError(f'Protocol receipt checksum mismatch: {path}')
    for source, checksum in body['source_files'].items():
        if sha256(source) != checksum:
            raise ValueError(f'Night ground-truth source changed: {source}')
    for image, checksum in body.get('image_sha256', {}).items():
        if sha256(image) != checksum:
            raise ValueError(f'Gardens original-image mapping bytes changed: {image}')
    refs = [Path(p).resolve(strict=True) for p in body['references']]
    queries = [Path(p).resolve(strict=True) for p in body['queries']]
    if set(refs) & set(queries):
        raise ValueError('Additional night protocol has overlapping query/gallery files')
    gt = checked_positives([np.asarray(row) for row in body['positives']], len(refs), len(queries))
    return Benchmark(body['name'], Path(body['root']), refs, queries, gt, body['protocol'],
                     {str(path.resolve()): sha256(path), **body['source_files']})


def sped():
    """All published SPED pairs; explicitly not a verified night-only subset."""
    root = Path('/home/admin123/github/datasets/SPED')
    files = [root / name for name in ('SPED_dbImages.npy', 'SPED_qImages.npy', 'SPED_gt.npy')]
    refs, queries = [np.load(p, allow_pickle=False) for p in files[:2]]
    gt = np.load(files[2], allow_pickle=True)
    return Benchmark('sped_mixed_illumination_diagnostic', root,
        [root / str(p) for p in refs], [root / str(p) for p in queries],
        checked_positives(gt, len(refs), len(queries)),
        dict(type='supplementary mixed-illumination diagnostic; NOT night-only',
             positive='published SPED_gt.npy', night_labels='no verified per-query night labels'),
        {str(p): sha256(p) for p in files})


EXTRA_BUILDERS = {
    'robotcar_public_night': lambda: load_protocol('configs/evaluation_sources/robotcar_public/night.json'),
    'robotcar_public_night_rain': lambda: load_protocol('configs/evaluation_sources/robotcar_public/night-rain.json'),
    'gardens_day_right': lambda: load_protocol('configs/evaluation_sources/gardenspoint/night_day_right.json'),
    'gardens_day_left': lambda: load_protocol('configs/evaluation_sources/gardenspoint/night_day_left.json'),
    'sped': sped,
}
