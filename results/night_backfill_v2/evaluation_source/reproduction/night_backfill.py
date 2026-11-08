"""Complete the retained three-method night matrix; fixed weights, no training."""
import argparse
import fcntl
import gc
import json
from pathlib import Path
import shutil
import signal

import numpy as np

from reproduction.benchmarks import build_benchmark, json_sha256, sha256
from reproduction.evaluate import (atomic_json, cache_complete, exact_topk, extract_features,
    feature_identity, load_inference_checkpoint, recall_at_k, validate_inference_checkpoint)
from reproduction.extra_night_benchmarks import EXTRA_BUILDERS
from reproduction.feature_io import BoundedFeatureRows
from reproduction.night_benchmarks import BUILDERS
from reproduction.night_suite import reuse_cache, verify_cache


def checked_metrics(pred, benchmark):
    pred = np.asarray(pred)
    if (pred.shape != (len(benchmark.queries), 10) or pred.dtype.kind not in 'iu'
            or pred.min() < 0 or pred.max() >= len(benchmark.references)):
        raise ValueError('Top-10 rows/indices do not match this night protocol')
    hits = np.asarray([[bool(np.isin(row[:k], pos).any()) for k in (1, 5, 10)]
                       for row, pos in zip(pred, benchmark.positives)])
    return recall_at_k(pred, benchmark.positives, (1, 5, 10)), hits


def reuse_result(benchmark, label, spec, teacher_sha, source, destination):
    """Reuse only identical weight/protocol results, auditing cache bytes and hits."""
    source = Path(source)
    path = source / 'summary.json'
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    old = summary.get('results', {}).get(benchmark.name, {}).get(label)
    if old is None:
        return None
    if (summary.get('status') not in ('complete', 'running') or summary.get('precision') != 'float32'
            or summary.get('tf32') is not False or summary.get('teacher_sha256') != teacher_sha
            or summary.get('protocols', {}).get(benchmark.name) != benchmark.describe()
            or summary.get('models', {}).get(label, {}).get('sha256') != spec['checkpoint_sha256']
            or old['checkpoint_sha256'] != spec['checkpoint_sha256']
            or old['teacher_sha256'] != teacher_sha):
        raise ValueError(f'Prior result differs in weights, precision or protocol: {path}/{benchmark.name}/{label}')
    predpath = source / benchmark.name / f'{label}_top10.npy'
    if sha256(predpath) != old['predictions_sha256']:
        raise ValueError('Existing night predictions failed checksum verification')
    config = json.loads(Path('configs/boq_b2.json').read_text())
    for kind, paths, model_sha, role in [
            ('gallery', benchmark.references, teacher_sha, 'gallery_teacher'),
            ('query', benchmark.queries, spec['checkpoint_sha256'], 'query_student')]:
        cache = Path(old[f'{kind}_cache']['path'])
        actual = verify_cache(cache, feature_identity(paths, config, model_sha, role))
        if actual != old[f'{kind}_cache']['feature_sha256']:
            raise ValueError('Existing row and feature-cache checksums differ')
    metrics, hits = checked_metrics(np.load(predpath), benchmark)
    if metrics != old['metrics_percent'] or old['query_count'] != len(benchmark.queries):
        raise ValueError('Existing row Recall differs from its actual predictions/ground truth')
    destination.mkdir(parents=True, exist_ok=True)
    newpred = destination / f'{label}_top10.npy'
    if not newpred.exists():
        shutil.copy2(predpath, newpred)
    if sha256(newpred) != old['predictions_sha256']:
        raise ValueError('Promoted predictions differ from audited source')
    hitpath = destination / f'{label}_hits.npy'
    np.save(hitpath, hits)
    return dict(old, predictions_sha256=sha256(newpred), hits_sha256=sha256(hitpath),
        evaluation_reuse=dict(source_summary=str(path.resolve()), source_summary_sha256=sha256(path),
                              source_metrics_sha256=json_sha256(old), recomputed_hits=True))


def report(out, summary):
    atomic_json(out / 'summary.json', summary)
    labels = list(summary['models'])
    names = [summary['models'][key]['name'] for key in labels]
    lines = ['# 三种保留方法：夜间评测补全', '',
        '固定已选 best 权重，测试不再选模；无新训练。每格为 R@1 / R@5 / R@10（%）。', '',
        '| 协议 | Query | ' + ' | '.join(names) + ' |',
        '| --- | ---: | ' + ' | '.join('---:' for _ in labels) + ' |']
    for dataset in summary['protocols']:
        rows = summary['results'].get(dataset, {})
        count = summary['protocols'][dataset]['queries']['count']
        values = []
        for label in labels:
            m = rows.get(label, {}).get('metrics_percent')
            values.append(' / '.join(f'{m[f"R@{k}"]:.4f}' for k in (1, 5, 10)) if m else '待完成')
        lines.append(f'| {dataset} | {count} | ' + ' | '.join(values) + ' |')
    lines += ['', 'Night 与 Night-rain 为两个独立条件；Gardens 的两行使用同一批夜间查询、不同白天图库。',
        'RobotCar GPS、RobotCar-Seasons public-train、DarkDriving 均为明确命名的诊断协议，不冒充官方定位 test。',
        'SPED 为补充的混合光照诊断，没有验证过的逐图夜间标签，不计入严格夜间数据集总数。',
        '同一协议下三方法共享同一冻结教师图库、原始测试图片、FP32 与关闭 TF32 的精确检索。',
        '已有结果仅在权重、输入顺序、真值、缓存完整字节和预测命中核验后复用。', '',
        '未能给出 Recall 的项目及原因：']
    for name, why in summary['coverage']['unavailable'].items():
        lines.append(f'- {name}：{why}')
    lines += ['', f'状态：{summary["status"]}。', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))


class ExtractionLogger:
    def __init__(self, logger, pair_step):
        self.logger, self.step = logger, pair_step

    def log(self, _images, **metrics):
        elapsed = next((v for k, v in metrics.items() if k.endswith('/elapsed_seconds')), None)
        images = next((v for k, v in metrics.items() if k.endswith('/images')), None)
        if elapsed and images is not None:
            metrics['eval/current_extraction_images_per_second'] = images / elapsed
        self.logger.log(self.step, **metrics)


def snapshot_identity(registry, descriptions):
    names = ['night_backfill.py', 'extra_night_benchmarks.py', 'prepare_night_backfill.py',
             'night_suite.py', 'night_benchmarks.py', 'feature_io.py', 'evaluate.py', 'benchmarks.py',
             'asym_model.py', 'data.py', 'path_resolver.py', 'monitoring.py', 'eval_launch.sh', 'night_finalize.py']
    sources = ['reproduction/' + n for n in names] + ['src/boq.py', 'src/backbones.py']
    return dict(registry_sha256=json_sha256(registry), manifests=descriptions,
                implementation_sha256={n: sha256(n) for n in sources})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registry', default='configs/night_backfill_v2.json')
    p.add_argument('--manifest-only', action='store_true')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--teacher-batch-size', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if args.batch_size <= 0 or args.teacher_batch_size <= 0 or args.workers < 0:
        p.error('Extraction batches must be positive and workers must be nonnegative')
    registry = json.loads(Path(args.registry).read_text())
    if set(registry['models']) != {'original', 'mixed', 'mixed_camera'}:
        raise ValueError('This requested backfill is restricted to the three retained methods')
    out, cache = Path(registry['output']), Path(registry['cache'])
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / '.evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    benches = [fn() for fn in [*BUILDERS.values(), *EXTRA_BUILDERS.values()]]
    descriptions = {b.name: b.describe() for b in benches}
    coverage = dict(included=list(descriptions) + ['tokyo105_night'],
        supplementary=['sped_mixed_illumination_diagnostic'],
        unavailable={
            'Aachen-Day-Night official night test': 'No public query pose / verified query-to-gallery retrieval positives',
            'RobotCar-Seasons official night / night-rain test': 'Test query poses withheld; public-train diagnostic evaluated separately',
            'NightCity / BDD10K / MUSES': 'Night images exist but no verified place-retrieval ground truth',
            'SVOX val': 'Local validation queries are Mapillary daytime views, with no verified night-query subset',
            'NCLT / 17Places': 'No complete local image/ground-truth bundle',
            'Boreas / Nordland': 'No verified night-query labels in the available local traversals'},
        training='Mixed+camera already completed 10 epochs / 9080 updates; retain its selected epoch 8, no duplicate training')
    atomic_json(out / 'manifests.json', descriptions)
    atomic_json(out / 'coverage.json', coverage)
    for name, row in descriptions.items():
        print(f'{name}: {row["references"]["count"]} refs / {row["queries"]["count"]} queries', flush=True)
    if args.manifest_only:
        return
    if sha256(registry['config']) != registry['config_sha256']:
        raise ValueError('Evaluation model configuration differs from pinned registry')
    config = json.loads(Path(registry['config']).read_text())
    teacher_sha = sha256(config['teacher_weights'])
    if teacher_sha != registry['teacher_sha256']:
        raise ValueError('Frozen teacher differs from pinned registry')
    for label, spec in registry['models'].items():
        if sha256(spec['checkpoint']) != spec['checkpoint_sha256'] or sha256(spec['source_summary']) != spec['source_summary_sha256']:
            raise ValueError(f'Selected weights or original test summary changed: {label}')
        state = load_inference_checkpoint(spec['checkpoint'])
        validate_inference_checkpoint(state, config, teacher_sha)
        del state
    identity = snapshot_identity(registry, descriptions)
    identity_path = out / 'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Immutable evaluation code/config/protocols changed; use a new output')
    atomic_json(identity_path, identity)
    for source in identity['implementation_sha256']:
        target = out / 'evaluation_source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    if (out / 'complete.json').exists():
        complete = json.loads((out / 'complete.json').read_text())
        for key, name in [('summary_sha256', 'summary.json'), ('report_sha256', 'REPORT.md')]:
            if complete[key] != sha256(out / name):
                raise ValueError('Completed backfill output changed')
        print('Backfill already complete; no duplicate inference started.', flush=True)
        return
    import torch
    import reproduction.evaluate as evaluation
    from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ
    from reproduction.monitoring import TrainingLogger
    from reproduction.night_finalize import _assert_same_protocol, tokyo_night_diagnostic
    signal.signal(signal.SIGINT, evaluation.request_stop)
    signal.signal(signal.SIGTERM, evaluation.request_stop)
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    free, total = torch.cuda.mem_get_info()
    budget = min(32 * 1024**3, free - 4 * 1024**3)
    if budget < 8 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory after 4 GiB reserve')
    torch.cuda.set_per_process_memory_fraction(budget / total)
    models = registry['models']
    summary = dict(status='running', operation=registry['operation'], models=models,
        precision='float32', tf32=False, teacher_sha256=teacher_sha, protocols=dict(descriptions), results={},
        coverage=coverage, resources=dict(student_batch=args.batch_size, teacher_batch=args.teacher_batch_size,
            workers=args.workers, cuda_allocator_budget_bytes=budget))
    tokyo = build_benchmark('tokyo247')
    tokyo_ref = Path(models['original']['source_summary']).parent / 'tokyo247_top10.npy'
    summary['protocols']['tokyo105_night'] = dict(queries={'count': 105},
        protocol='trusted N labels in the original NetVLAD 315-query subset, full gallery')
    pairs, total_pairs = 0, (len(benches) + 1) * len(models)
    with TrainingLogger(config=dict(operation=registry['operation'], registry=args.registry), total_steps=total_pairs) as logger:
        for label, spec in models.items():
            src = Path(spec['source_summary'])
            old = json.loads(src.read_text())
            predicted = src.parent / 'tokyo247_top10.npy'
            _assert_same_protocol(old['datasets']['tokyo247'], tokyo.describe(), 'tokyo247')
            if (old['checkpoint_sha256'] != spec['checkpoint_sha256']
                    or old['teacher_sha256'] != teacher_sha
                    or sha256(predicted) != old['datasets']['tokyo247']['predictions_sha256']):
                raise ValueError('Tokyo fixed prediction/source identity mismatch')
            diagnostic = tokyo_night_diagnostic(tokyo, tokyo_ref, predicted)
            if not diagnostic['available'] or diagnostic['num_queries'] != 105:
                raise ValueError('Tokyo trusted night protocol is unavailable')
            summary['results'].setdefault('tokyo105_night', {})[label] = dict(
                metrics_percent=diagnostic['final_percent'], query_count=105,
                checkpoint_sha256=spec['checkpoint_sha256'], existing_evaluation=True,
                source_summary=str(src), source_summary_sha256=sha256(src), diagnostic=diagnostic)
            pairs += 1
            logger.log(pairs, **{f'eval/tokyo105_night/{label}/{k}': v for k, v in diagnostic['final_percent'].items()})
        report(out, summary)
        for b in benches:
            dest = out / b.name
            for label, spec in models.items():
                if evaluation.STOP:
                    raise SystemExit(130)
                row = None
                for source in registry['reusable_suites']:
                    row = reuse_result(b, label, spec, teacher_sha, source, dest)
                    if row is not None:
                        break
                if row is None:
                    gi = feature_identity(b.references, config, teacher_sha, 'gallery_teacher')
                    qi = feature_identity(b.queries, config, spec['checkpoint_sha256'], 'query_student')
                    gallery = cache / 'gallery' / json_sha256(gi) / 'features.npy'
                    query = cache / 'query' / json_sha256(qi) / 'features.npy'
                    if not cache_complete(gallery, gi):
                        teacher = GalleryBoQ(config).float().cuda().eval()
                        extract_features(teacher, b.references, gallery, gi, config, 'cuda',
                            args.teacher_batch_size, args.workers, ExtractionLogger(logger, pairs), b.name + '/gallery')
                        del teacher
                        gc.collect(); torch.cuda.empty_cache()
                    if evaluation.STOP:
                        raise SystemExit(130)
                    if not cache_complete(query, qi):
                        state = load_inference_checkpoint(spec['checkpoint'])
                        model = EfficientViTBoQ(config, pretrained=False)
                        model.load_state_dict(state['model'], strict=True)
                        del state
                        model = model.float().cuda().eval()
                        extract_features(model, b.queries, query, qi, config, 'cuda',
                            args.batch_size, args.workers, ExtractionLogger(logger, pairs), b.name + '/' + label)
                        del model
                        gc.collect(); torch.cuda.empty_cache()
                    gallery_sha, query_sha = verify_cache(gallery, gi), verify_cache(query, qi)
                    with BoundedFeatureRows(gallery) as refs, BoundedFeatureRows(query) as queries:
                        info = {}
                        pred, _ = exact_topk(refs, queries, 10, 64, 4096, 'cuda',
                            reference_bank='auto', cuda_reserve_gib=4., search_info=info)
                    metrics, hits = checked_metrics(pred, b)
                    dest.mkdir(parents=True, exist_ok=True)
                    np.save(dest / f'{label}_top10.npy', pred)
                    np.save(dest / f'{label}_hits.npy', hits)
                    row = dict(metrics_percent=metrics, query_count=len(b.queries), gallery_count=len(b.references),
                        checkpoint_sha256=spec['checkpoint_sha256'], teacher_sha256=teacher_sha,
                        protocol_sha256=descriptions[b.name]['manifest_sha256'],
                        gallery_cache=dict(path=str(gallery.resolve()), feature_sha256=gallery_sha),
                        query_cache=dict(path=str(query.resolve()), feature_sha256=query_sha), search_memory=info,
                        predictions_sha256=sha256(dest / f'{label}_top10.npy'),
                        hits_sha256=sha256(dest / f'{label}_hits.npy'), existing_evaluation=False)
                atomic_json(dest / f'{label}_metrics.json', row)
                summary['results'].setdefault(b.name, {})[label] = row
                pairs += 1
                logger.log(pairs, **{f'eval/{b.name}/{label}/{k}': v for k, v in row['metrics_percent'].items()},
                    **{'eval/completed_pairs': pairs, 'eval/remaining_pairs': total_pairs - pairs,
                       'eval/reused_result': int('evaluation_reuse' in row),
                       'resources/cuda_peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3})
                report(out, summary)
                print(f'{pairs}/{total_pairs} {b.name}/{label}: {row["metrics_percent"]}', flush=True)
                gc.collect(); torch.cuda.empty_cache()
    if pairs != total_pairs or any(set(summary['results'].get(k, {})) != set(models) for k in summary['protocols']):
        raise AssertionError('Cannot mark incomplete method-by-protocol matrix complete')
    summary['status'] = 'complete'
    report(out, summary)
    atomic_json(out / 'complete.json', dict(status='complete', pairs=pairs, identity_sha256=json_sha256(identity),
        summary_sha256=sha256(out / 'summary.json'), report_sha256=sha256(out / 'REPORT.md')))


if __name__ == '__main__':
    main()
