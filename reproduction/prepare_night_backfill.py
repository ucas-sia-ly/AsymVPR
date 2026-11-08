"""Freeze independently audited public-night protocols; no model training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

from reproduction.benchmarks import json_sha256, sha256
from reproduction.evaluate import atomic_json

SOURCES = Path('configs/evaluation_sources')
WEATHER = Path('/home/admin123/github/WeatherVPR')
ROBOTCAR = Path('/home/admin123/github/WORKSPACE/dataset/RobotCar-Seasons')
GARDENS = Path('artifacts/benchmark_sources/gardens_original')


def sealed_json(path, body):
    value = dict(body, record_sha256=json_sha256(body))
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'Refusing to replace an existing different protocol: {path}')
    if not path.exists():
        atomic_json(path, value)
    return value


def robotcar():
    # Reuse the user's previously audited strict 49-model COLMAP parser ONCE.
    # Evaluation uses the frozen JSON below, without importing WeatherVPR.
    sys.path.insert(0, str(WEATHER / 'src'))
    from weather_vpr.inference.robotcar import load_robotcar_public_night
    parser = WEATHER / 'src/weather_vpr/inference/robotcar.py'
    for condition in ('night', 'night-rain'):
        data = load_robotcar_public_night(ROBOTCAR, condition)
        protocol = data.protocol
        files = {str(Path(protocol['metadata_path']).resolve()): protocol['metadata_sha256'],
                 str(parser.resolve()): sha256(parser)}
        files.update({str((ROBOTCAR / row['path']).resolve()): row['sha256']
                      for row in protocol['colmap_pose_models']})
        body = dict(schema=1, name='robotcar_seasons_public_' + condition.replace('-', '_'),
            root=str(ROBOTCAR), references=[str(x.path.resolve()) for x in data.references],
            queries=[str(x.path.resolve()) for x in data.queries], positives=data.positives,
            protocol=protocol, source_files=files)
        path = SOURCES / 'robotcar_public' / f'{condition}.json'
        sealed_json(path, body)
        print(f'{path}: {len(data.references)} refs, {len(data.queries)} queries', flush=True)


def gardens():
    archive = Path('artifacts/benchmark_sources/GardensPointWalking.zip')
    if hashlib.md5(archive.read_bytes()).hexdigest() != '1284fe9747dd5cb3e6c3358b37ae361f':
        raise ValueError('Gardens Point archive differs from original Zenodo release MD5')
    sequences = {name: {} for name in ('day_left', 'day_right', 'night_right')}
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            p = Path(name)
            if p.parent.name not in sequences or p.suffix.lower() != '.jpg':
                continue
            frame = int(p.stem.removeprefix('Image'))
            if not 0 <= frame < 200 or p.name != f'Image{frame:03d}.jpg':
                raise ValueError('Unexpected original Gardens sequence member')
            data = z.read(name)
            digest = hashlib.sha256(data).hexdigest()
            target = GARDENS / p.parent.name / p.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and sha256(target) != digest:
                raise ValueError('Existing canonical Gardens image differs from original ZIP')
            if not target.exists():
                target.write_bytes(data)
            sequences[p.parent.name][frame] = (str(target.resolve()), digest)
    if any(set(seq) != set(range(200)) for seq in sequences.values()):
        raise ValueError('Gardens must match all original 600 images and all three 200-frame orders')
    if len({item[0] for seq in sequences.values() for item in seq.values()}) != 600:
        raise ValueError('Gardens original-to-local mapping is not bijective')
    gt = SOURCES / 'gardenspoint/GardensPointWalking_ground_truth_new.npy'
    raw = np.load(gt, allow_pickle=True)
    if raw.shape != (200, 2) or [int(x[0]) for x in raw] != list(range(200)):
        raise ValueError('Unexpected author Gardens Point ground-truth layout')
    positives = []
    clipped = 0
    for _, values in raw:
        row = np.asarray(values, dtype=np.int64)
        clipped += int(((row < 0) | (row >= 200)).sum())
        positives.append(row[(row >= 0) & (row < 200)].tolist())
    for side in ('day_left', 'day_right'):
        protocol = dict(type='Gardens Point night retrieval; VPR-Bench ordinal positives',
            split='all 200 original night_right queries', gallery=side,
            positive='published GardensPointWalking ground_truth_new.npy; intersect valid 200-reference IDs',
            invalid_published_boundary_indices_clipped=clipped,
            mapping='All 600 images extracted byte-for-byte from the MD5-verified original Zenodo ZIP; canonical names retained',
            canonical_order='Image000.jpg through Image199.jpg in each original sequence',
            source_urls=['https://zenodo.org/records/4590133',
                'https://github.com/MubarizZaffar/VPR-Bench/blob/77ad30f3684476717188c67d7f3f66be19f832af/datasets/GardensPointWalking/ground_truth_new.npy'],
            limitation='Published frame-tolerance retrieval protocol, not metric-distance ground truth')
        body = dict(schema=1, name='gardenspoint_night_' + side, root=str(GARDENS),
            references=[sequences[side][i][0] for i in range(200)],
            queries=[sequences['night_right'][i][0] for i in range(200)], positives=positives,
            protocol=protocol, source_files={str(archive.resolve()): sha256(archive), str(gt.resolve()): sha256(gt)},
            image_sha256={path: digest for seq in sequences.values() for path, digest in seq.values()})
        path = SOURCES / 'gardenspoint' / f'night_{side}.json'
        sealed_json(path, body)
        print(f'{path}: all 600 source image bytes verified, 200 refs / 200 night queries', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('robotcar', 'gardens', 'all'), default='all')
    args = parser.parse_args()
    if args.dataset in ('robotcar', 'all'):
        robotcar()
    if args.dataset in ('gardens', 'all'):
        gardens()


if __name__ == '__main__':
    main()
