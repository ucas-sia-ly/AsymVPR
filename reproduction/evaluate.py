"""FP32 asymmetric VPR evaluation with bounded-memory, resumable feature caches.

python -m reproduction.evaluate --dataset msls --checkpoint checkpoints/boq_b2_best.pt
Use --manifest-only to validate all paths/protocols without loading a model.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from reproduction.benchmarks import (DATASETS, PAPER, build_benchmark, json_sha256,
                                      ordered_path_identity, sha256)

TRANSFORM = 'ToTensor/Resize-bicubic-antialias/ImageNet-normalize'
STOP = False


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def request_stop(*_):
    global STOP
    STOP = True
    print('Stop requested; committing the current feature batch.', flush=True)


def load_inference_checkpoint(path):
    """Allow only NumPy scalar metadata in otherwise weights-only checkpoints."""
    import torch
    # The published local best.pt contains np.float64 Recall values, not pickle objects.
    scalar = np._core.multiarray.scalar if hasattr(np, '_core') else np.core.multiarray.scalar
    with torch.serialization.safe_globals([scalar, np.dtype, type(np.dtype('float64'))]):
        state = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or not isinstance(state.get('model'), dict):
        raise ValueError('Expected inference checkpoint with model state_dict')
    return state


def validate_inference_checkpoint(state, config, teacher_sha):
    """A compatible student must be paired with its exact training gallery."""
    checkpoint_config = state.get('config', {})
    if not isinstance(checkpoint_config, dict):
        raise ValueError('Checkpoint model config must be a dictionary')
    for key in ('student_name', 'descriptor_dim', 'proj_channels', 'num_queries', 'num_layers', 'image_size'):
        if checkpoint_config.get(key) != config[key]:
            raise ValueError(f'Checkpoint model config mismatch: {key}')
    if config.get('precision') != 'float32':
        raise ValueError('Evaluation requires config precision=float32')
    expected = dict(teacher_sha256=teacher_sha, image_size=config['image_size'],
                    descriptor_dim=config['descriptor_dim'], transform=TRANSFORM, precision='float32')
    provenance = state.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('Checkpoint training provenance must be a dictionary')
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f'Checkpoint training provenance mismatch: {key}')


def feature_identity(paths, config, model_sha, role):
    return dict(schema=1, role=role, model_sha256=model_sha,
                image_size=config['image_size'], descriptor_dim=config['descriptor_dim'],
                transform=TRANSFORM, precision='float32', tf32=False,
                paths=ordered_path_identity(paths))


def open_feature_cache(target, identity):
    """Only committed rows may be resumed; a cache without its identity is rejected."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta_path, state_path = target.with_suffix('.provenance.json'), target.with_suffix('.progress.json')
    expected = (identity['paths']['count'], identity['descriptor_dim'])
    if target.exists() or meta_path.exists() or state_path.exists():
        if not (target.exists() and meta_path.exists() and state_path.exists()):
            raise RuntimeError(f'Incomplete cache metadata: {target}; use another cache directory')
        meta, progress = json.loads(meta_path.read_text()), json.loads(state_path.read_text())
        if meta != identity or progress.get('identity_sha256') != json_sha256(identity):
            raise RuntimeError(f'Feature-cache provenance mismatch: {target}')
        features = np.load(target, mmap_mode='r+')
        if features.shape != expected or features.dtype != np.float32:
            raise ValueError(f'Feature-cache shape/dtype mismatch: {target}')
        rows = progress['rows']
        if not isinstance(rows, int) or not 0 <= rows <= expected[0]:
            raise ValueError('Invalid committed feature row count')
        if rows == expected[0] and progress.get('feature_sha256') != sha256(target):
            raise RuntimeError(f'Completed feature-cache checksum mismatch: {target}')
    else:
        features = np.lib.format.open_memmap(target, mode='w+', dtype='float32', shape=expected)
        atomic_json(meta_path, identity)
        rows = 0
        atomic_json(state_path, dict(rows=0, total=expected[0], identity_sha256=json_sha256(identity)))
    return features, rows


def cache_complete(target, identity):
    if not Path(target).exists():
        return False
    array, rows = open_feature_cache(target, identity)
    result = rows == len(array)
    del array
    return result


def extract_features(model, paths, target, identity, config, device, batch_size, workers, logger, phase):
    import torch
    from reproduction.data import Images, loader
    features, initial = open_feature_cache(target, identity)
    n = len(paths)
    if initial == n:
        del features
        return
    model.eval()
    start_time = time.monotonic()
    dataset = Images('/', np.asarray([str(p) for p in paths]), config['image_size'], start=initial)
    for images, indexes in loader(dataset, batch_size, workers):
        indices = indexes.numpy()
        with torch.inference_mode():
            embeddings = model(images.to(device=device, dtype=torch.float32, non_blocking=True))
        values = embeddings.detach().cpu().numpy()
        if values.dtype != np.float32 or values.shape != (len(indices), identity['descriptor_dim']):
            raise ValueError(f'Unexpected embedding dtype/shape: {values.dtype}/{values.shape}')
        norms = np.linalg.norm(values, axis=1)
        if not np.isfinite(values).all() or not np.allclose(norms, 1., atol=5e-4, rtol=5e-4):
            raise FloatingPointError(f'Nonfinite or non-unit embeddings in {phase}')
        if not np.array_equal(indices, np.arange(initial, initial + len(indices))):
            raise RuntimeError('Feature extractor violated deterministic manifest order')
        features[initial:initial + len(indices)] = values
        initial += len(indices)
        features.flush()
        progress = dict(rows=initial, total=n, identity_sha256=json_sha256(identity))
        if initial == n:
            progress['feature_sha256'] = sha256(target)
        atomic_json(Path(target).with_suffix('.progress.json'), progress)
        elapsed = time.monotonic() - start_time
        logger.log(initial, **{f'eval/{phase}/images': initial,
                               f'eval/{phase}/elapsed_seconds': elapsed})
        print(f'{phase}: {initial}/{n}', flush=True)
        if STOP:
            raise SystemExit(130)
    del features


def resident_reference_budget(shape, free_bytes, query_chunk, reference_chunk, reserve_gib=4.):
    """Budget an FP32 CUDA bank plus bounded search/validation scratch space."""
    if len(shape) != 2 or min(shape) <= 0 or min(query_chunk, reference_chunk) <= 0:
        raise ValueError('Invalid reference-bank dimensions or chunk sizes')
    if not 2. <= reserve_gib <= 4.:
        raise ValueError('CUDA reference-bank reserve must be between 2 and 4 GiB')
    nrefs, dim = shape
    bank_bytes = int(nrefs * dim * 4)
    chunk = min(nrefs, reference_chunk)
    # FP32 query + FP32 similarity block + one bool finite-check block.
    workspace_bytes = int(query_chunk * dim * 4 + query_chunk * chunk * 4 + chunk * dim)
    reserve_bytes = int(reserve_gib * 1024 ** 3)
    required_bytes = bank_bytes + workspace_bytes + reserve_bytes
    return dict(bank_bytes=bank_bytes, workspace_budget_bytes=workspace_bytes,
                reserve_bytes=reserve_bytes, free_bytes=int(free_bytes),
                required_bytes=required_bytes, fits=required_bytes <= free_bytes)


def cuda_reference_bank(references, device, query_chunk, reference_chunk, reserve_gib=4.):
    """Copy a bank once in bounded host chunks, or explicitly fall back on OOM.

    No full NumPy copy is created. Input finiteness is checked exactly once per
    uploaded chunk. Other errors (including invalid descriptors) propagate.
    """
    import torch
    cuda_device = torch.device(device)
    if cuda_device.type != 'cuda':
        raise ValueError('Resident reference bank requires a CUDA device')
    free_bytes, total_bytes = torch.cuda.mem_get_info(cuda_device)
    budget = resident_reference_budget(references.shape, free_bytes, query_chunk,
                                       reference_chunk, reserve_gib)
    info = dict(mode='cuda_streamed', device=str(cuda_device), total_bytes=int(total_bytes),
                **budget)
    if not budget['fits']:
        info['fallback_reason'] = 'insufficient free CUDA memory for bank, workspace and reserve'
        return None, info
    bank = None
    try:
        bank = torch.empty(references.shape, dtype=torch.float32, device=cuda_device)
        with torch.inference_mode():
            for start in range(0, len(references), reference_chunk):
                stop = min(start + reference_chunk, len(references))
                source = torch.from_numpy(np.array(references[start:stop], copy=True))
                bank[start:stop].copy_(source)
                if not torch.isfinite(bank[start:stop]).all():
                    raise FloatingPointError('Nonfinite reference feature')
                if STOP:
                    raise SystemExit(130)
    except torch.cuda.OutOfMemoryError:
        # Free only this process's incomplete bank; memory races may invalidate
        # the initial free-memory estimate. The original chunked path remains exact.
        bank = None
        gc.collect()
        torch.cuda.empty_cache()
        info['fallback_reason'] = 'CUDA OOM while allocating or filling reference bank'
        return None, info
    info['mode'] = 'cuda_resident'
    return bank, info


def exact_topk(references, queries, k=10, query_chunk=64, reference_chunk=8192,
               device='cpu', progress=None, reference_bank='auto', cuda_reserve_gib=4.,
               search_info=None):
    """Exhaustive inner-product search, bounded by query_chunk x reference_chunk.

    Equal scores use ascending reference index for deterministic chunk-independent ties.
    Inputs must be FP32, unit-normalized descriptors. No approximate index is used.
    CUDA may hold references once when actual free memory leaves 2-4 GiB reserve;
    reference_bank='off' retains bounded streaming. CPU behavior is unchanged.
    """
    import torch
    if references.ndim != 2 or queries.ndim != 2 or references.shape[1] != queries.shape[1]:
        raise ValueError('Incompatible feature matrices')
    if references.dtype != np.float32 or queries.dtype != np.float32:
        raise ValueError('Exact retrieval requires FP32 descriptors')
    if not (0 < k <= len(references)) or query_chunk <= 0 or reference_chunk <= 0:
        raise ValueError('Invalid top-k or chunk size')
    if reference_bank not in ('auto', 'off'):
        raise ValueError('reference_bank must be auto or off')
    cuda_device = torch.device(device)
    bank = None
    info = dict(mode='cpu_streamed' if cuda_device.type == 'cpu' else 'cuda_streamed',
                reference_bank_option=reference_bank)
    if cuda_device.type == 'cuda' and reference_bank == 'auto':
        bank, details = cuda_reference_bank(references, cuda_device, query_chunk,
                                             reference_chunk, cuda_reserve_gib)
        info.update(details)
        print(f'Search reference bank: {info}', flush=True)
    if search_info is not None:
        search_info.clear()
        search_info.update(info)
    predictions = np.empty((len(queries), k), dtype=np.int64)
    scores_out = np.empty((len(queries), k), dtype=np.float32)
    for qs in range(0, len(queries), query_chunk):
        q = torch.from_numpy(np.array(queries[qs:qs + query_chunk], copy=True)).to(device)
        if not torch.isfinite(q).all():
            raise FloatingPointError('Nonfinite query feature')
        best_scores = np.empty((len(q), 0), dtype=np.float32)
        best_indices = np.empty((len(q), 0), dtype=np.int64)
        for rs in range(0, len(references), reference_chunk):
            if bank is None:
                r = torch.from_numpy(np.array(references[rs:rs + reference_chunk], copy=True)).to(device)
                if not torch.isfinite(r).all():
                    raise FloatingPointError('Nonfinite reference feature')
            else:
                r = bank[rs:rs + reference_chunk]
            with torch.inference_mode():
                block = (q @ r.T).cpu().numpy()
            # Sorting the bounded block also specifies exact behavior at ties.
            order = np.argsort(-block, axis=1, kind='stable')[:, :k]
            block_scores = np.take_along_axis(block, order, axis=1)
            candidate_indices = np.concatenate((best_indices, order + rs), axis=1)
            candidate_scores = np.concatenate((best_scores, block_scores), axis=1)
            order = np.lexsort((candidate_indices, -candidate_scores), axis=1)[:, :k]
            best_indices = np.take_along_axis(candidate_indices, order, axis=1)
            best_scores = np.take_along_axis(candidate_scores, order, axis=1)
        predictions[qs:qs + len(q)] = best_indices
        scores_out[qs:qs + len(q)] = best_scores
        if progress:
            progress(qs + len(q), len(queries))
        if STOP:
            raise SystemExit(130)
    return predictions, scores_out


def recall_at_k(predictions, positives, ks=(1, 5, 10)):
    if len(predictions) != len(positives) or not len(positives):
        raise ValueError('Query/ground-truth count mismatch or empty benchmark')
    if max(ks) > predictions.shape[1]:
        raise ValueError('Not enough predictions for requested recalls')
    return {f'R@{k}': float(100 * np.mean([np.isin(row[:k], gt).any()
             for row, gt in zip(predictions, positives)])) for k in ks}


def legacy_msls_reference(config, benchmark, identity, path):
    """Explicit legacy reuse: reconstruct original extraction order and disclose limits."""
    from reproduction.data import msls
    path = Path(path)
    parent = path.parent
    source_identity = json.loads((parent / 'provenance.json').read_text())
    for key in ('teacher_sha256', 'image_size', 'descriptor_dim', 'transform', 'precision'):
        expected = identity['model_sha256'] if key == 'teacher_sha256' else identity[key]
        if source_identity.get(key) != expected:
            raise RuntimeError(f'Legacy MSLS provenance mismatch: {key}')
    progress = json.loads(path.with_suffix('.progress.json').read_text())
    original_config = dict(config, msls_root=str(benchmark.root))
    paths, nrefs, positives = msls(original_config)
    original_paths = [(benchmark.root / str(p)).resolve() for p in paths[:nrefs]]
    if original_paths != benchmark.references or any(not np.array_equal(a, b) for a, b in zip(positives, benchmark.positives)):
        raise ValueError('Legacy MSLS manifest/positive order differs from evaluation')
    features = np.load(path, mmap_mode='r')
    if features.dtype != np.float32 or features.shape != (len(paths), identity['descriptor_dim']):
        raise ValueError('Legacy MSLS feature shape/dtype mismatch')
    if progress != dict(rows=len(paths), total=len(paths)):
        raise ValueError('Legacy MSLS extraction is incomplete')
    metadata_files = [Path(p) for p in benchmark.source_files]
    if any(p.stat().st_mtime_ns > path.stat().st_mtime_ns for p in metadata_files):
        raise ValueError('Legacy MSLS metadata newer than feature cache; order cannot be verified')
    note = dict(source=str(path.resolve()), feature_sha256=sha256(path),
                verified=['teacher checksum', 'transform/precision/dimension', 'completion',
                          'reconstructed original msls() manifest and positive order', 'metadata predates cache'],
                limitations=['legacy provenance did not record input-order hash or TF32 setting'],
                provenance=source_identity)
    return features[:nrefs], note


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_b2.json')
    parser.add_argument('--checkpoint', default='checkpoints/boq_b2_best.pt')
    parser.add_argument('--dataset', nargs='+', choices=(*DATASETS, 'all'), default=['msls'])
    parser.add_argument('--dataset-root', default='/home/admin123/github/datasets')
    parser.add_argument('--metadata-root', default='/home/admin123/third_party/salad/datasets')
    parser.add_argument('--roots-json', help='JSON object mapping benchmark names to image roots')
    parser.add_argument('--nordland-protocol', choices=['paper', 'salad'], default='paper')
    parser.add_argument('--cache-dir', default='artifacts/evaluation_cache')
    parser.add_argument('--output-dir', default='runs/evaluation')
    parser.add_argument('--legacy-msls-cache', help='Explicitly reuse old teacher cache, with disclosed provenance limits')
    parser.add_argument('--manifest-only', action='store_true')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--query-chunk', type=int, default=64)
    parser.add_argument('--reference-chunk', type=int, default=8192)
    parser.add_argument('--reference-bank', choices=['auto', 'off'], default='auto')
    parser.add_argument('--cuda-reserve-gib', type=float, default=4.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cpu-threads', type=int, default=6)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    names = list(DATASETS) if 'all' in args.dataset else list(dict.fromkeys(args.dataset))
    roots = json.loads(Path(args.roots_json).read_text()) if args.roots_json else {}
    benches = [build_benchmark(name, args.dataset_root, args.metadata_root, roots,
                              args.nordland_protocol) for name in names]
    descriptions = {b.name: b.describe() for b in benches}
    output, cache = Path(args.output_dir), Path(args.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'manifests.json', descriptions)
    for b in benches:
        print(f'{b.name}: {len(b.references)} references, {len(b.queries)} queries; {b.protocol}', flush=True)
    if args.manifest_only:
        return
    import torch
    from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ
    from reproduction.monitoring import TrainingLogger
    torch.set_num_threads(args.cpu_threads)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    checkpoint_sha, teacher_sha = sha256(args.checkpoint), sha256(config['teacher_weights'])
    # Always validate the source pair, even when every feature cache is complete.
    checkpoint_state = load_inference_checkpoint(args.checkpoint)
    validate_inference_checkpoint(checkpoint_state, config, teacher_sha)
    identities = {b.name: {
        'gallery': feature_identity(b.references, config, teacher_sha, 'gallery_teacher'),
        'query': feature_identity(b.queries, config, checkpoint_sha, 'query_student'),
    } for b in benches}
    legacy = {}
    if args.legacy_msls_cache:
        if 'msls' not in names:
            raise ValueError('--legacy-msls-cache requires MSLS in --dataset')
        b = next(b for b in benches if b.name == 'msls')
        _, legacy['msls'] = legacy_msls_reference(config, b, identities['msls']['gallery'], args.legacy_msls_cache)
    log_dir = Path(os.environ.get('TRAIN_TERMINAL_RUN_DIR', str(output)))
    with TrainingLogger(run_dir=log_dir, config=dict(vars(args), precision='float32', tf32=False)) as logger:
        # Each phase owns exactly one model; no teacher/student coexistence.
        for role, constructor in [('gallery', lambda: GalleryBoQ(config)),
                                  ('query', lambda: EfficientViTBoQ(config, pretrained=False))]:
            needed = [b for b in benches if not (role == 'gallery' and b.name in legacy)
                      and not cache_complete(cache / b.name / f'{role}.npy', identities[b.name][role])]
            if not needed:
                continue
            model = constructor()
            if role == 'query':
                model.load_state_dict(checkpoint_state['model'], strict=True)
            model = model.float().to(args.device).eval()
            for b in needed:
                paths = b.references if role == 'gallery' else b.queries
                extract_features(model, paths, cache / b.name / f'{role}.npy', identities[b.name][role],
                                 config, args.device, args.batch_size, args.workers, logger, f'{b.name}/{role}')
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del checkpoint_state
        summary_path = output / 'summary.json'
        summary = dict(schema=1, checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_sha256=checkpoint_sha,
                       teacher_sha256=teacher_sha, config_sha256=sha256(args.config),
                       precision='float32', tf32=False, search='exact chunked inner product, deterministic index ties',
                       datasets={})
        for step, b in enumerate(benches, 1):
            if b.name in legacy:
                refs, _ = legacy_msls_reference(config, b, identities[b.name]['gallery'], args.legacy_msls_cache)
            else:
                refs = np.load(cache / b.name / 'gallery.npy', mmap_mode='r')
            queries = np.load(cache / b.name / 'query.npy', mmap_mode='r')
            start = time.monotonic()
            search_info = {}
            predictions, scores = exact_topk(refs, queries, min(10, len(refs)), args.query_chunk,
                                             args.reference_chunk, args.device,
                                             lambda done, total: print(f'{b.name}/search: {done}/{total}', flush=True),
                                             reference_bank=args.reference_bank,
                                             cuda_reserve_gib=args.cuda_reserve_gib, search_info=search_info)
            ks = tuple(k for k in (1, 5, 10) if k <= predictions.shape[1])
            metrics = recall_at_k(predictions, b.positives, ks)
            np.save(output / f'{b.name}_top10.npy', predictions)
            np.save(output / f'{b.name}_scores.npy', scores)
            paper = {f'R@{k}': v for k, v in PAPER[b.name].items()}
            result = dict(**descriptions[b.name], metrics_percent=metrics,
                          paper_table1_percent=paper,
                          difference_percentage_points={key: metrics[key] - val for key, val in paper.items()},
                          search_seconds=time.monotonic() - start,
                          search_memory=search_info,
                          gallery_cache=legacy.get(b.name) or dict(path=str((cache / b.name / 'gallery.npy').resolve()),
                            feature_sha256=sha256(cache / b.name / 'gallery.npy'), provenance=identities[b.name]['gallery']),
                          query_cache=dict(path=str((cache / b.name / 'query.npy').resolve()),
                            feature_sha256=sha256(cache / b.name / 'query.npy'), provenance=identities[b.name]['query']),
                          predictions_sha256=sha256(output / f'{b.name}_top10.npy'))
            if b.name == 'nordland' and args.nordland_protocol != 'paper':
                result['paper_comparison_note'] = 'Ground-truth tolerance differs from paper; difference is diagnostic only'
            summary['datasets'][b.name] = result
            atomic_json(output / f'{b.name}_metrics.json', result)
            atomic_json(summary_path, summary)
            logger.log(step, **{f'eval/{b.name}/{key}': value for key, value in metrics.items()})
            print(json.dumps(dict(dataset=b.name, **metrics), ensure_ascii=False), flush=True)
            del refs, queries


if __name__ == '__main__':
    main()
