"""Night adaptation development evaluation in the unchanged BoQ gallery space.

Synthetic validation holds out NEW night augmentations by place; the initial
student has already seen their daytime GSV places. SVOX development uses the
existing geographically purged TRAIN-night manifest, never test-night metrics.
No function starts training or silently creates a new data selection.
"""
from contextlib import contextmanager
import gc
import json
from pathlib import Path

import numpy as np

from reproduction.benchmarks import (build_benchmark, checked_positives,
    json_sha256, ordered_path_identity, radius_positives, sha256)
from reproduction.evaluate import (TRANSFORM, atomic_json, cache_complete,
    exact_topk, extract_features, feature_identity, open_feature_cache, recall_at_k)


class IndexedFeatures:
    """An ordered, lazy view of mmap rows; retrieval copies only requested chunks."""
    def __init__(self, values, indices):
        self.values = values
        self.indices = np.asarray(indices, dtype=np.int64)
        if self.indices.ndim != 1 or not len(self.indices):
            raise ValueError('Empty or non-vector feature index')
        if self.indices.min() < 0 or self.indices.max() >= len(values):
            raise ValueError('Feature index outside fixed teacher bank')
        self.shape = (len(self.indices), values.shape[1])
        self.ndim, self.dtype = 2, values.dtype

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        return self.values[self.indices[key]]


def synthetic_layout(all_rows, val_rows):
    """Freeze all source-day distractors; every same-place source is a positive."""
    if not all_rows or not val_rows:
        raise ValueError('Synthetic evaluation needs a nonempty snapshot and validation split')
    lookup, reference_map, reference_labels = {}, {}, []
    place_labels, label_places = {}, {}
    for index, row in enumerate(all_rows):
        name, source, label = str(Path(row['night_path']).resolve()), int(row['source_index']), int(row['label'])
        place = (str(row['city']), int(row['place_id']))
        if name in lookup or source < 0 or label < 0:
            raise ValueError('Duplicate night identity or invalid teacher row/label')
        if (place in place_labels and place_labels[place] != label
                or label in label_places and label_places[label] != place):
            raise ValueError('Inconsistent place/label identity in frozen snapshot')
        place_labels[place], label_places[label] = label, place
        lookup[name] = index
        if source in reference_map:
            if reference_labels[reference_map[source]] != label:
                raise ValueError('One source teacher row has conflicting place labels')
        else:
            reference_map[source] = len(reference_labels)
            reference_labels.append(label)
    query_indices, val_labels, val_names = [], [], set()
    for row in val_rows:
        name = str(Path(row['night_path']).resolve())
        if name not in lookup or name in val_names or all_rows[lookup[name]] != row:
            raise ValueError('Validation row is duplicated or differs from frozen all-row snapshot')
        val_names.add(name)
        query_indices.append(lookup[name])
        val_labels.append(int(row['label']))
    # Splitting variants of one place would leak the adaptation validation.
    heldout_labels = set(val_labels)
    if any(int(row['label']) in heldout_labels and str(Path(row['night_path']).resolve()) not in val_names
           for row in all_rows):
        raise ValueError('Synthetic train/validation split leaks a place')
    labels = np.asarray(reference_labels)
    positives = [np.flatnonzero(labels == label) for label in val_labels]
    checked_positives(positives, len(labels), len(query_indices))
    return dict(reference_indices=list(reference_map), query_indices=query_indices,
                positives=positives, reference_labels=reference_labels,
                num_heldout_places=len(heldout_labels))


def _root(config):
    return Path(config.get('eval_output_dir', str(Path(config['output_dir']) / 'evaluation')))


def _teacher_sha(config):
    return sha256(config['teacher_weights'])


_VERIFIED = {}


def _fixed_features(path, identity):
    """Hash completed caches once per unchanged file/sidecar identity per process."""
    path = Path(path)
    companions = [path, path.with_suffix('.provenance.json'), path.with_suffix('.progress.json')]
    signature = (json_sha256(identity), tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in companions))
    key = str(path.resolve())
    if _VERIFIED.get(key) != signature:
        array, rows = open_feature_cache(path, identity)
        if rows != len(array):
            raise RuntimeError(f'Incomplete teacher evaluation cache: {path}')
        del array
        _VERIFIED[key] = signature
    return np.load(path, mmap_mode='r')


def _feature_receipt(path):
    progress = json.loads(Path(path).with_suffix('.progress.json').read_text())
    if not isinstance(progress.get('feature_sha256'), str) or len(progress['feature_sha256']) != 64:
        raise ValueError('Completed evaluation feature cache lacks a SHA256 receipt')
    return dict(path=str(Path(path).resolve()), feature_sha256=progress['feature_sha256'])


def _gsv_gallery(config, all_rows, layout):
    base = Path(config.get('base_cache', 'artifacts/gallery_cache'))
    provenance = json.loads((base / 'provenance.json').read_text())
    expected = dict(teacher_sha256=_teacher_sha(config), image_size=config['image_size'],
                    descriptor_dim=config['descriptor_dim'], transform=TRANSFORM, precision='float32')
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError('Source-day teacher cache does not match the frozen BoQ teacher/preprocessing')
    paths = np.load(base / 'paths.npy', mmap_mode='r')
    values = np.load(base / 'train_features.npy', mmap_mode='r')
    progress = json.loads((base / 'train_features.progress.json').read_text())
    if (values.shape != (len(paths), config['descriptor_dim']) or values.dtype != np.float32
            or progress.get('rows') != len(paths) or progress.get('total') != len(paths)):
        raise ValueError('Incomplete or invalid original GSV teacher feature bank')
    real_root = Path(config['gsv_root'])
    for row in all_rows:
        index = int(row['source_index'])
        if not 0 <= index < len(paths):
            raise ValueError('Synthetic source_index outside original day bank')
        raw = paths[index]
        raw = raw.decode() if isinstance(raw, (bytes, np.bytes_)) else str(raw)
        if (real_root / raw).resolve() != Path(row['source_path']).resolve():
            raise ValueError('Synthetic exact source does not match original teacher manifest order')
    return IndexedFeatures(values, layout['reference_indices']), dict(
        source=str((base / 'train_features.npy').resolve()), provenance=provenance,
        selected_source_indices_sha256=json_sha256(layout['reference_indices']),
        original_feature_file_size=(base / 'train_features.npy').stat().st_size,
        original_feature_file_mtime_ns=(base / 'train_features.npy').stat().st_mtime_ns,
        limitation='Inherited original GSV bank has no complete feature-file checksum; source mapping and teacher identity checked')


@contextmanager
def _evaluation_mode(model):
    import torch
    states = [(module, module.training) for module in model.modules()]
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    model.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        yield
    finally:
        for module, training in states:
            module.training = training
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32


def _queries(model, paths, config, device):
    import torch
    from reproduction.data import Images, loader
    result = np.empty((len(paths), config['descriptor_dim']), dtype=np.float32)
    batches = loader(Images('/', np.asarray([str(p) for p in paths]), config['image_size']),
                     config.get('eval_batch_size', 64), config.get('num_workers', 2))
    done = 0
    with _evaluation_mode(model), torch.inference_mode(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for images, indices in batches:
            indices = indices.numpy()
            if not np.array_equal(indices, np.arange(done, done + len(indices))):
                raise ValueError('Live query extraction order differs from fixed validation manifest')
            values = model(images.to(device=device, dtype=torch.float32, non_blocking=True)).cpu().numpy()
            if (values.dtype != np.float32 or values.shape != (len(indices), config['descriptor_dim'])
                    or not np.isfinite(values).all()
                    or not np.allclose(np.linalg.norm(values, axis=1), 1., atol=5e-4, rtol=5e-4)):
                raise FloatingPointError('Invalid live validation descriptors')
            result[done:done + len(indices)] = values
            done += len(indices)
    if done != len(paths):
        raise ValueError('Live validation extractor returned an incomplete query set')
    return result


def _search(references, queries, positives, config, device):
    import torch
    tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        predictions, _ = exact_topk(references, queries, min(10, len(references)),
            config.get('eval_query_chunk', 64), config.get('eval_reference_chunk', 8192), device,
            reference_bank=config.get('eval_reference_bank', 'auto'),
            cuda_reserve_gib=config.get('cuda_reserve_gib', 4.))
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32
    ks = tuple(k for k in (1, 5, 10) if k <= predictions.shape[1])
    return recall_at_k(predictions, positives, ks), predictions


def _publish(name, result, predictions, config, epoch, step, logger):
    tag = f'epoch_{epoch:02d}' if epoch is not None else f'step_{step:08d}'
    target = _root(config) / name
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / f'{tag}_top10.npy', predictions)
    result.update(epoch=epoch, step=step, precision='float32', tf32=False,
                  predictions_sha256=sha256(target / f'{tag}_top10.npy'))
    atomic_json(target / f'{tag}.json', result)
    if logger is not None:
        logger.log(step, **{f'val/{name}/{key}': val for key, val in result['metrics_percent'].items()})
    return result


def _teacher_baseline(name, refs, teacher_queries, positives, definition, config, device):
    path = _root(config) / name / 'teacher_baseline.json'
    identity = json_sha256(definition)
    if path.is_file():
        row = json.loads(path.read_text())
        body = {key: value for key, value in row.items() if key != 'result_sha256'}
        if row.get('result_sha256') != json_sha256(body):
            raise ValueError('Fixed teacher baseline result checksum mismatch')
        if row.get('definition_sha256') != identity:
            raise ValueError(f'Fixed teacher baseline protocol changed: {path}')
        if not row.get('metrics_percent') or any(not 0 <= value <= 100 for value in row['metrics_percent'].values()):
            raise ValueError('Invalid cached fixed-teacher baseline metrics')
        return row['metrics_percent']
    metrics, _ = _search(refs, teacher_queries, positives, config, device)
    body = dict(definition_sha256=identity, definition=definition, metrics_percent=metrics)
    atomic_json(path, dict(**body, result_sha256=json_sha256(body)))
    return metrics


def evaluate_synthetic(model, config, *, device='cuda', epoch=None, step=0, logger=None):
    from reproduction.night_data import load_records
    all_rows, val_rows = load_records(config, 'all'), load_records(config, 'val')
    layout = synthetic_layout(all_rows, val_rows)
    references, source = _gsv_gallery(config, all_rows, layout)
    teacher_path = Path(config['night_teacher_cache'])
    teacher_identity = feature_identity([Path(r['night_path']) for r in all_rows], config,
                                        _teacher_sha(config), 'night_teacher')
    teacher_values = _fixed_features(teacher_path, teacher_identity)
    teacher_queries = IndexedFeatures(teacher_values, layout['query_indices'])
    definition = dict(protocol='new synthetic augmentations held out by place; all frozen source-day references',
        snapshot_sha256=json_sha256(all_rows), ground_truth_sha256=json_sha256([p.tolist() for p in layout['positives']]),
        source_gallery=source, night_teacher_cache_identity=teacher_identity,
        night_teacher_cache_receipt=_feature_receipt(teacher_path),
        scope='Initial checkpoint already saw heldout GSV daytime places; not unseen-location evaluation',
        semantic_status='Pending verification is preserved; teacher consistency is not semantic approval')
    teacher_metrics = _teacher_baseline('synthetic_night', references, teacher_queries,
                                        layout['positives'], definition, config, device)
    query_values = _queries(model, [r['night_path'] for r in val_rows], config, device)
    metrics, predictions = _search(references, query_values, layout['positives'], config, device)
    return _publish('synthetic_night', dict(metrics_percent=metrics, teacher_night_percent=teacher_metrics,
        num_references=len(references), num_queries=len(val_rows),
        heldout_places=layout['num_heldout_places'], definition=definition),
        predictions, config, epoch, step, logger)


def _msls(config):
    benchmark = build_benchmark('msls', roots={'msls': config['msls_root']})
    path = Path(config.get('msls_gallery_cache', 'results/evaluation_cache/msls/gallery.npy'))
    identity = feature_identity(benchmark.references, config, _teacher_sha(config), 'gallery_teacher')
    return benchmark, _fixed_features(path, identity), identity


def evaluate_msls_guard(model, config, *, device='cuda', epoch=None, step=0, logger=None):
    benchmark, references, identity = _msls(config)
    queries = _queries(model, benchmark.queries, config, device)
    metrics, predictions = _search(references, queries, benchmark.positives, config, device)
    return _publish('msls_guard', dict(metrics_percent=metrics, num_references=len(references),
        num_queries=len(queries), definition=benchmark.describe(), teacher_gallery_identity=identity,
        teacher_gallery_receipt=_feature_receipt(config.get('msls_gallery_cache', 'results/evaluation_cache/msls/gallery.npy'))),
        predictions, config, epoch, step, logger)


def _safe_relative(root, value):
    value = Path(value)
    if value.is_absolute() or '..' in value.parts:
        raise ValueError('SVOX frozen manifest paths must be relative to explicit svox_root')
    path = (root / value).resolve(strict=True)
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('SVOX manifest escapes its explicit image root')
    return path


def _coords(paths):
    return np.asarray([[float(p.name.split('@')[1]), float(p.name.split('@')[2])] for p in paths], np.float64)


def load_svox_dev(config):
    """Validate the existing frozen train-night/25 m/50 m-purge protocol in full."""
    from scipy.spatial import cKDTree
    manifest = Path(config['svox_dev_manifest'])
    receipt = manifest.parent / 'protocol.json'
    protocol, rows = json.loads(receipt.read_text()), json.loads(manifest.read_text())
    if (protocol.get('manifest_sha256') != sha256(manifest)
            or protocol.get('role') != 'natural_weather_validation'
            or protocol.get('source') != 'official SVOX train split'
            or protocol.get('positive_rule') != 'UTM distance <= 25 m to train gallery'
            or protocol.get('exclusion') != 'drop queries within 50 m of any SVOX test/val gallery or query image'):
        raise ValueError('SVOX development receipt/hash/25 m positives/50 m purge fields differ')
    root = Path(config.get('svox_root', '/home/admin123/github/datasets/svox')).resolve()
    refs = rows['references']
    nights = [row for row in rows['queries'] if Path(row['id']).parent.as_posix() == 'images/train/queries_night']
    if (len(refs) != protocol['gallery_images'] or len(nights) != protocol['queries_per_domain']['night']
            or not nights or len({r['id'] for r in refs}) != len(refs)):
        raise ValueError('SVOX frozen development counts or reference IDs differ')
    if any(r['id'] != r['path'] or Path(r['id']).parent.as_posix() != 'images/train/gallery' for r in refs):
        raise ValueError('SVOX dev references must be the complete frozen train gallery')
    if any(r['id'] != r['path'] for r in nights) or len({r['id'] for r in nights}) != len(nights):
        raise ValueError('SVOX night query identity/path mismatch')
    references = [_safe_relative(root, r['path']) for r in refs]
    queries = [_safe_relative(root, r['path']) for r in nights]
    current_gallery = sorted(p.resolve() for p in (root / 'images/train/gallery').glob('*.jpg'))
    if sorted(references) != current_gallery:
        raise ValueError('Frozen SVOX development gallery no longer equals complete train gallery')
    ref_index = {row['id']: i for i, row in enumerate(refs)}
    positives = checked_positives([np.array([ref_index[p] for p in r['positives']], np.int64)
                                  for r in nights], len(refs), len(nights))
    audited = radius_positives(_coords(references), _coords(queries), 25.)
    if any(not np.array_equal(a, b) for a, b in zip(positives, audited)):
        raise ValueError('Frozen SVOX night positives differ from exact 25 m UTM coordinates')
    excluded = []
    for split in ('val', 'test'):
        for directory in sorted((root / 'images' / split).iterdir()):
            if directory.is_dir() and (directory.name == 'gallery' or directory.name.startswith('queries')):
                excluded.extend(sorted(directory.glob('*.jpg')))
    if not excluded:
        raise ValueError('Cannot verify SVOX dev geographic purge without val/test metadata')
    distances, _ = cKDTree(_coords(excluded)).query(_coords(queries))
    if not np.isfinite(distances).all() or np.any(distances < 50.):
        raise ValueError('SVOX dev query violates 50 m exclusion from val/test images')
    definition = dict(manifest=str(manifest.resolve()), manifest_sha256=sha256(manifest),
        protocol_receipt_sha256=sha256(receipt), protocol=protocol, actual_root=str(root),
        reference_identity=ordered_path_identity(references), query_identity=ordered_path_identity(queries),
        ground_truth_sha256=json_sha256([p.tolist() for p in positives]),
        purged_against_ordered_paths_sha256=json_sha256([str(p.resolve()) for p in excluded]),
        minimum_distance_to_val_test_m=float(distances.min()),
        scope='Oxford same-city, same-camera target-domain development; not unseen-domain test')
    return dict(references=references, queries=queries, positives=positives, definition=definition)


def _svox_paths(config):
    default_cache = Path(config.get('eval_cache_dir', str(Path(config['input_dir']).parent / 'cache/eval')))
    directory = Path(config.get('svox_dev_cache', str(default_cache / 'svox_dev')))
    return directory / 'gallery.npy', directory / 'night_teacher.npy'


def prepare_eval_caches(config, logger, teacher=None, device='cuda'):
    """Prepare fixed SVOX teacher targets before loading the trainable student."""
    from reproduction.asym_model import GalleryBoQ
    import torch
    _, _, msls_identity = _msls(config)
    if not config.get('svox_dev_manifest'):
        return dict(msls_gallery='verified', msls_identity=msls_identity, svox_dev='not configured')
    data = load_svox_dev(config)
    targets = _svox_paths(config)
    identities = [feature_identity(data[role], config, _teacher_sha(config), 'gallery_teacher')
                  for role in ('references', 'queries')]
    pending = [not cache_complete(path, identity) for path, identity in zip(targets, identities)]
    owned_teacher = teacher is None
    if any(pending):
        if owned_teacher:
            teacher = GalleryBoQ(config).float().to(device).eval()
        with _evaluation_mode(teacher):
            for needed, role, target, identity in zip(pending, ('references', 'queries'), targets, identities):
                if needed:
                    extract_features(teacher, data[role], target, identity, config, device,
                        config.get('extract_batch_size', 128), config.get('num_workers', 2), logger,
                        f'svox_dev_teacher/{role}')
        if owned_teacher:
            del teacher
            gc.collect()
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
    atomic_json(_root(config) / 'svox_dev_definition.json', data['definition'])
    return dict(msls_gallery='verified', svox_dev=data['definition'],
                teacher_caches=[str(p.resolve()) for p in targets])


def evaluate_svox_dev(model, config, *, device='cuda', epoch=None, step=0, logger=None):
    data = load_svox_dev(config)
    gallery_path, night_path = _svox_paths(config)
    gallery_identity = feature_identity(data['references'], config, _teacher_sha(config), 'gallery_teacher')
    night_identity = feature_identity(data['queries'], config, _teacher_sha(config), 'gallery_teacher')
    references = _fixed_features(gallery_path, gallery_identity)
    teacher_queries = _fixed_features(night_path, night_identity)
    definition = dict(**data['definition'], teacher_gallery_identity=gallery_identity,
                       teacher_night_identity=night_identity,
                       teacher_gallery_receipt=_feature_receipt(gallery_path),
                       teacher_night_receipt=_feature_receipt(night_path))
    teacher_metrics = _teacher_baseline('svox_night_dev', references, teacher_queries,
                                        data['positives'], definition, config, device)
    queries = _queries(model, data['queries'], config, device)
    metrics, predictions = _search(references, queries, data['positives'], config, device)
    return _publish('svox_night_dev', dict(metrics_percent=metrics, teacher_night_percent=teacher_metrics,
        num_references=len(references), num_queries=len(queries), definition=definition),
        predictions, config, epoch, step, logger)
