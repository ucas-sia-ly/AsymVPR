"""Strict benchmark manifests; metadata order is part of the evaluation protocol."""
from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


PAPER = {
    'msls': {1: 92.3, 5: 95.9}, 'pitts250k': {1: 95.4, 5: 98.5},
    'tokyo247': {1: 92.7, 5: 96.5}, 'nordland': {1: 74.6, 5: 88.3},
    'amstertime': {1: 48.6, 5: 71.3},
}
DATASETS = tuple(PAPER)
DEFAULT_DATA = Path('/home/admin123/github/datasets')
SALAD_METADATA = Path('/home/admin123/third_party/salad/datasets')
NORDLAND_RELEASE = Path('/home/admin123/github/icra/third_party/VLAD-BUFF/logs/nordland_official_protocol/data/Nordland')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def ordered_path_identity(paths):
    """Record order plus input file size/mtime; fail on missing/duplicate images."""
    names, stats = [], []
    for value in paths:
        p = Path(value).resolve(strict=True)
        if not p.is_file():
            raise ValueError(f'Image is not a regular file: {p}')
        stat = p.stat()
        names.append(str(p))
        stats.append([str(p), stat.st_size, stat.st_mtime_ns])
    if len(set(names)) != len(names):
        raise ValueError('Duplicate image paths within one benchmark role')
    return dict(count=len(names), ordered_paths_sha256=json_sha256(names),
                input_file_stats_sha256=json_sha256(stats))


def checked_positives(positives, nrefs, nqueries):
    if len(positives) != nqueries:
        raise ValueError(f'{len(positives)} ground-truth rows for {nqueries} queries')
    result = []
    for i, row in enumerate(positives):
        raw = np.asarray(row)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError(f'Ground truth row {i} must be a 1-D integer array')
        row = np.unique(raw.astype(np.int64))
        if not len(row) or row[0] < 0 or row[-1] >= nrefs:
            raise ValueError(f'Invalid or empty ground truth for query {i}')
        result.append(row)
    return result


def radius_positives(ref_coords, query_coords, radius=25.):
    """UTM coordinates in metres, inclusive Euclidean radius."""
    refs, queries = np.asarray(ref_coords, np.float64), np.asarray(query_coords, np.float64)
    if refs.ndim != 2 or refs.shape[1] != 2 or queries.ndim != 2 or queries.shape[1] != 2:
        raise ValueError('Expected N x 2 UTM coordinates')
    if not np.isfinite(refs).all() or not np.isfinite(queries).all():
        raise ValueError('Nonfinite UTM coordinates')
    return [np.flatnonzero(np.square(refs - q).sum(1) <= radius ** 2) for q in queries]


def frame_positives(nrefs, query_frames, tolerance=3):
    if tolerance < 0:
        raise ValueError('Negative frame tolerance')
    result = []
    for frame in query_frames:
        frame = int(frame)
        if not 0 <= frame < nrefs:
            raise ValueError(f'Query frame {frame} outside reference sequence')
        result.append(np.arange(max(0, frame - tolerance), min(nrefs, frame + tolerance + 1)))
    return result


@dataclass
class Benchmark:
    name: str
    root: Path
    references: list
    queries: list
    positives: list
    protocol: dict
    source_files: dict

    def describe(self):
        refs = ordered_path_identity(self.references)
        queries = ordered_path_identity(self.queries)
        gt_hash = json_sha256([p.tolist() for p in self.positives])
        value = dict(dataset=self.name, root=str(self.root.resolve()),
                     references=refs, queries=queries, protocol=self.protocol,
                     ground_truth_sha256=gt_hash, metadata_sha256=self.source_files)
        value['manifest_sha256'] = json_sha256(value)
        return value


def build_benchmark(name, dataset_root=DEFAULT_DATA, metadata_root=SALAD_METADATA,
                    roots=None, nordland_protocol='paper'):
    """Resolve every image; no query filtering beyond the published MSLS qIdx."""
    from reproduction.path_resolver import resolve_paths
    if name not in DATASETS:
        raise ValueError(f'Unknown benchmark {name}')
    roots = roots or {}
    base, meta = Path(dataset_root), Path(metadata_root)
    preferred_nordland = NORDLAND_RELEASE if NORDLAND_RELEASE.is_dir() else base / 'Nordland'
    root = Path(roots.get(name, {
        'msls': base / 'msls_val', 'pitts250k': base / 'Pittsburgh/Pittsburgh',
        'tokyo247': base / 'tokyo247', 'nordland': preferred_nordland,
        'amstertime': base / 'amstertime',
    }[name]))
    source = {}

    def load(filename, subdir='', objects=False):
        candidates = [root / filename, meta / subdir / filename]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f'Missing metadata: {candidates}')
        source[str(path.resolve())] = sha256(path)
        return np.load(path, allow_pickle=objects)

    if name == 'msls':
        refs = load('msls_val_dbImages.npy', 'msls_val')
        all_queries = load('msls_val_qImages.npy', 'msls_val')
        qidx = load('msls_val_qIdx.npy', 'msls_val')
        if qidx.ndim != 1 or not np.issubdtype(qidx.dtype, np.integer) or len(np.unique(qidx)) != len(qidx):
            raise ValueError('Invalid MSLS qIdx')
        if qidx.min() < 0 or qidx.max() >= len(all_queries):
            raise ValueError('MSLS qIdx out of bounds')
        queries = all_queries[qidx]
        positives = load('msls_val_pIdx.npy', 'msls_val', True)
        protocol = dict(split='validation', positives='official pIdx, 25 m geographical threshold',
                        query_selection='official qIdx, queries with known positives',
                        all_query_count=len(all_queries), selected_query_count=len(queries))
    elif name == 'pitts250k':
        refs = load('pitts250k_test_dbImages.npy', 'Pittsburgh')
        queries = load('pitts250k_test_qImages.npy', 'Pittsburgh')
        positives = load('pitts250k_test_gt.npy', 'Pittsburgh', True)
        protocol = dict(split='pitts250k_test', positives='official GSV-Cities/SALAD ground truth, 25 m')
        mat_path = root / 'datasets/pitts250k_test.mat'
        if mat_path.is_file():
            from scipy.io import loadmat
            from scipy.spatial import cKDTree
            original = loadmat(mat_path, squeeze_me=True, struct_as_record=False)['dbStruct']
            if (float(original.posDistThr) != 25. or not np.array_equal(refs, original.dbImageFns)
                    or not np.array_equal(queries, np.char.add('queries_real/', original.qImageFns))):
                raise ValueError('Pittsburgh NPY/MAT order or 25 m threshold mismatch')
            audited = cKDTree(original.utmDb.T).query_ball_point(original.utmQ.T, 25.)
            if len(audited) != len(positives) or any(not np.array_equal(np.sort(a), np.sort(b))
                                                     for a, b in zip(audited, positives)):
                raise ValueError('Pittsburgh NPY ground truth does not match source MAT at 25 m')
            source[str(mat_path.resolve())] = sha256(mat_path)
            protocol['source_mat_audit'] = 'all image IDs/order and all 25 m positive sets match exactly'
    elif name == 'tokyo247':
        refs = load('tokyo247_dbImages.npy', 'Tokyo247')
        queries = load('tokyo247_qImages.npy', 'Tokyo247')
        coords = lambda paths: [[float(str(p).split('@')[1]), float(str(p).split('@')[2])] for p in paths]
        positives = radius_positives(coords(refs), coords(queries), 25.)
        protocol = dict(split='Tokyo24/7, published 315-query subset',
                        positives='UTM coordinates encoded in official paths; Euclidean distance <= 25 m',
                        resolution='original VGA, deterministic resize to configured input size')
    elif name == 'nordland':
        refs = load('Nordland_dbImages.npy', 'Nordland')
        queries = load('Nordland_qImages.npy', 'Nordland')
        official = load('Nordland_gt.npy', 'Nordland', True)
        query_frames = [int(Path(str(p)).stem) * 10 for p in queries]
        if nordland_protocol == 'paper':
            positives = frame_positives(len(refs), query_frames, 3)
        elif nordland_protocol == 'salad':
            positives = official
        else:
            raise ValueError('Nordland protocol must be paper or salad')
        protocol = dict(split='VPR-Bench sampled summer queries / winter references',
                        positives='reference sequence frame +/- 3 inclusive' if nordland_protocol == 'paper'
                        else 'published SALAD ground truth (reference frame +/- 1)',
                        query_frame='query ordinal * 10', protocol_variant=nordland_protocol,
                        published_gt_differing_rows=sum(not np.array_equal(a, b) for a, b in zip(positives, official)))
        protocol['image_layout'] = 'original VPR-Bench release ref/query' if (root / 'ref').is_dir() else 'native downloader, reversed season roles'
        release_provenance = root.parent.parent / 'provenance.json'
        if (root / 'ref').is_dir() and release_provenance.is_file():
            source[str(release_provenance.resolve())] = sha256(release_provenance)
    else:
        refs = load('amstertime_dbImages.npy', 'AmsterTime')
        queries = load('amstertime_qImages.npy', 'AmsterTime')
        pairs_path = root / 'image_pairs.csv'
        source[str(pairs_path.resolve())] = sha256(pairs_path)
        with pairs_path.open(newline='') as f:
            pairs = list(csv.DictReader(f))
        label = lambda path: int(str(path).split('@')[7])
        reference_index = {label(p): i for i, p in enumerate(refs)}
        if len(reference_index) != len(refs):
            raise ValueError('Duplicate AmsterTime reference IDs')
        pair_map = {}
        for row in pairs:
            pair_map.setdefault(int(row['old_num']), []).append(reference_index[int(row['new_num'])])
        positives = [np.array(pair_map[label(p)], dtype=np.int64) for p in queries]
        protocol = dict(split='all 1231 published pairs', references='new images', queries='old images',
                        positives='image_pairs.csv old_num -> new_num; no coordinate proxy')

    positive_rows = checked_positives(positives, len(refs), len(queries))
    references = resolve_paths(name, root, refs)
    query_paths = resolve_paths(name, root, queries)
    return Benchmark(name, root, references, query_paths, positive_rows, protocol, source)
