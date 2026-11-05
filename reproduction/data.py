"""Deterministic on-disk image manifest and bounded-memory gallery features."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def transform(size):
    # Matches the published BoQ tensor/bicubic/ImageNet evaluation transform.
    return T.Compose([T.ToTensor(), T.Resize((size, size),
        interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.Normalize([.485, .456, .406], [.229, .224, .225])])


def manifest(root, cache):
    import pandas as pd
    root, cache = Path(root), Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    csvs = sorted((root / 'Dataframes').glob('*.csv'))
    if not csvs:
        raise FileNotFoundError(f'No GSV metadata under {root}')
    digest = hashlib.sha256(str(root.resolve()).encode())
    for p in csvs:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    signature = digest.hexdigest()
    meta_path = cache / 'manifest.json'
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta['signature'] != signature:
            raise RuntimeError('Dataset changed: choose a new cache directory')
        return meta
    n = sum(len(pd.read_csv(p, usecols=['place_id'])) for p in csvs)
    paths = np.lib.format.open_memmap(cache / 'paths.npy', mode='w+', dtype='S256', shape=(n,))
    labels = np.lib.format.open_memmap(cache / 'labels.npy', mode='w+', dtype='int32', shape=(n,))
    offsets, keys, i, missing = [0], [], 0, []
    for p in csvs:
        df = pd.read_csv(p)
        for place, rows in df.groupby('place_id', sort=True):
            keys.append([p.stem, int(place)])
            for row in rows.itertuples(index=False):
                name = (f'{row.city_id}_{int(row.place_id):07d}_{int(row.year):04d}_'
                        f'{int(row.month):02d}_{int(row.northdeg):03d}_{row.lat}_{row.lon}_{row.panoid}.jpg')
                relative = f'Images/{row.city_id}/{name}'
                # Pandas decimal parsing can differ in the last digit from filenames.
                # Resolve against the published per-city filename index, never drop a row.
                if not (root / relative).is_file():
                    if 'file_index' not in locals() or indexed_city != row.city_id:
                        indexed_city = row.city_id
                        file_index = {}
                        for f in (root / 'Images' / row.city_id).glob('*.jpg'):
                            fields = f.name.split('_')
                            key = (fields[1], fields[2], fields[3], fields[4], '_'.join(fields[7:]))
                            if key in file_index:
                                raise ValueError(f'Ambiguous image key in {row.city_id}: {key}')
                            file_index[key] = f.name
                    key = (f'{int(row.place_id):07d}', f'{int(row.year):04d}',
                           f'{int(row.month):02d}', f'{int(row.northdeg):03d}', f'{row.panoid}.jpg')
                    actual = file_index.get(key)
                    if actual is None:
                        missing.append(relative)
                    else:
                        relative = f'Images/{row.city_id}/{actual}'
                if len(relative.encode()) > 256:
                    raise ValueError('Image path exceeds manifest slot')
                paths[i], labels[i] = relative.encode(), len(keys) - 1
                i += 1
            offsets.append(i)
    if missing:
        atomic_json(cache / 'missing_images.json', missing)
        raise FileNotFoundError(f'{len(missing)} images missing; see missing_images.json')
    paths.flush(); labels.flush()
    np.save(cache / 'offsets.npy', np.array(offsets, dtype=np.int64))
    atomic_json(cache / 'place_keys.json', keys)
    meta = dict(signature=signature, num_images=n, num_places=len(keys),
                cities=[p.stem for p in csvs], root=str(root.resolve()))
    atomic_json(meta_path, meta)
    return meta


class Images(Dataset):
    def __init__(self, root, paths, size, start=0):
        self.root, self.paths, self.start = Path(root), paths, start
        self.transform = transform(size)

    def __len__(self):
        return len(self.paths) - self.start

    def __getitem__(self, index):
        i = index + self.start
        name = self.paths[i]
        if isinstance(name, (bytes, np.bytes_)):
            name = name.decode()
        with Image.open(self.root / str(name)) as image:
            x = self.transform(image.convert('RGB'))
        return x, i


def loader(dataset, batch_size, workers, shuffle=False, generator=None):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=True, prefetch_factor=2 if workers else None,
        persistent_workers=False, generator=generator)


def place_statistics(features, offsets, out):
    """Arithmetic mean and unbiased diagonal sample covariance, one place at a time."""
    n, dim = len(offsets) - 1, features.shape[1]
    means = np.lib.format.open_memmap(Path(out) / 'centroids.npy', mode='w+', dtype='float32', shape=(n, dim))
    variances = np.lib.format.open_memmap(Path(out) / 'variances.npy', mode='w+', dtype='float32', shape=(n, dim))
    for j in range(n):
        block = np.asarray(features[offsets[j]:offsets[j+1]], dtype=np.float32)
        means[j] = block.mean(axis=0)
        variances[j] = block.var(axis=0, ddof=1) if len(block) > 1 else 0.
    means.flush(); variances.flush()


def msls(config):
    root = Path(config['msls_root'])
    refs = np.load(root / 'msls_val_dbImages.npy')
    queries = np.load(root / 'msls_val_qImages.npy')
    # Official MSLS evaluation uses only queries with at least one positive (740).
    qidx = np.load(root / 'msls_val_qIdx.npy')
    positives = np.load(root / 'msls_val_pIdx.npy', allow_pickle=True)
    paths = np.concatenate([refs, queries[qidx]])
    paths = np.array([p.removeprefix('train_val/') for p in paths])
    if len(qidx) != len(positives) or len(np.unique(qidx)) != len(qidx):
        raise ValueError('Invalid MSLS query/positive correspondence')
    for p in paths:
        if not (root / p).is_file():
            raise FileNotFoundError(root / p)
    return paths, len(refs), positives
