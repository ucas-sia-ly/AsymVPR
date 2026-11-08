"""Fixed final evaluation after night training, with verified read-only gallery reuse.

``finalize(config)`` is called only after training completes and its GPU objects
are released. This module never trains, edits original results, or extracts a
new teacher gallery. A subprocess owns the final student evaluation.
"""
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from reproduction.benchmarks import (DATASETS, build_benchmark, json_sha256, sha256)
from reproduction.evaluate import (TRANSFORM, atomic_json, feature_identity, load_inference_checkpoint,
                                   validate_inference_checkpoint)

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_SUMMARY = PROJECT / 'results/evaluation/summary.json'
SOURCE_CACHE = PROJECT / 'results/evaluation_cache'
TOKYO_SOURCE = Path('/home/admin123/github/LoQI-VPR/configs/evaluation_sources/tokyo247_netvlad_v100.source.json')
TOKYO_MAT = TOKYO_SOURCE.with_suffix('').with_suffix('.mat')


def _sealed(value):
    return dict(**value, record_sha256=json_sha256(value))


def _read_sealed(path):
    value = json.loads(Path(path).read_text())
    body = {key: item for key, item in value.items() if key != 'record_sha256'}
    if value.get('record_sha256') != json_sha256(body):
        raise ValueError(f'Finalization receipt checksum mismatch: {path}')
    return body


def _assert_same_protocol(before, after, dataset):
    for key in ('references', 'queries', 'protocol', 'ground_truth_sha256', 'metadata_sha256'):
        if before.get(key) != after.get(key):
            raise ValueError(f'{dataset}: final and initial evaluation protocols differ: {key}')


def _metrics(row):
    values = row.get('metrics_percent', {})
    if set(values) != {'R@1', 'R@5', 'R@10'} or any(
            not isinstance(v, (int, float)) or not np.isfinite(v) or not 0 <= v <= 100
            for v in values.values()):
        raise ValueError('Expected finite R@1/5/10 percentages')
    return values


def _validate_mixed_domain(config):
    if config.get('query_domain', 'mixed') != 'mixed':
        raise ValueError('Only mixed-query adaptation is supported')


def _dev_identity(definition):
    """Compare the same SVOX inputs/teacher bytes across cache directory names."""
    value = json.loads(json.dumps(definition))
    value.pop('manifest', None)  # The metadata bytes themselves remain SHA-bound.
    for key in ('teacher_gallery_receipt', 'teacher_night_receipt'):
        if key in value:
            value[key].pop('path', None)
    return value


def _mixed_reference(config, initial, dev_definition):
    """Load a SHA-pinned, sealed completed mixed run without changing its files."""
    keys = ('mixed_reference_summary', 'mixed_reference_comparison')
    if not any(config.get(key) for key in keys):
        return None
    if not all(config.get(key) for key in keys):
        raise ValueError('Mixed reference needs both summary and sealed comparison')
    summary_path, comparison_path = (Path(config[key]).resolve(strict=True) for key in keys)
    for key, path in zip(keys, (summary_path, comparison_path)):
        expected = config.get(f'{key}_sha256')
        if not isinstance(expected, str) or len(expected) != 64 or sha256(path) != expected:
            raise ValueError(f'Pinned mixed-reference SHA256 mismatch: {key}')
    if summary_path.parent != comparison_path.parent:
        raise ValueError('Mixed reference summary/comparison must belong to the same final evaluation')
    root = summary_path.parent.parent
    receipt_path, completed_path = root / 'finalization.json', root / 'train/completed.json'
    receipt, comparison = _read_sealed(receipt_path), _read_sealed(comparison_path)
    summary, completed = json.loads(summary_path.read_text()), json.loads(completed_path.read_text())
    if (receipt.get('status') != 'complete' or receipt.get('comparison') != comparison
            or receipt.get('identity_sha256') != json_sha256(receipt.get('identity'))):
        raise ValueError('Mixed reference comparison differs from its completed finalization identity')
    for path in (summary_path, comparison_path):
        if receipt['outputs_sha256'].get(str(path)) != sha256(path):
            raise ValueError('Mixed reference output is not bound by its finalization receipt')
    checkpoint = Path(summary['checkpoint']).resolve(strict=True)
    if checkpoint != (root / 'train/best.pt').resolve(strict=True):
        raise ValueError('Mixed reference summary points outside its selected best checkpoint')
    checkpoint_sha = sha256(checkpoint)
    state = load_inference_checkpoint(checkpoint)
    validate_inference_checkpoint(state, config, initial['teacher_sha256'])
    if (summary.get('checkpoint_sha256') != checkpoint_sha
            or receipt['identity'].get('best_sha256') != checkpoint_sha
            or receipt['identity'].get('config_sha256') != json_sha256(state['config'])
            or receipt['identity'].get('training_completed_sha256') != sha256(completed_path)
            or completed.get('status') != 'completed' or completed.get('config') != state['config']
            or comparison.get('training_config') != state['config']
            or state['config'].get('query_domain', 'mixed') != 'mixed'
            or any(state.get(key) != completed['best'].get(key) for key in ('epoch', 'step'))
            or comparison.get('selected') != completed['best']
            or comparison.get('adapted') != completed['best'].get('adapted')):
        raise ValueError('Mixed reference selected-checkpoint/config/comparison identity mismatch')
    if (summary.get('teacher_sha256') != initial['teacher_sha256']
            or summary.get('precision') != 'float32' or summary.get('tf32') is not False):
        raise ValueError('Mixed reference changed teacher or evaluation precision')
    if compare_summaries(initial, summary) != comparison['five_benchmark_comparison']:
        raise ValueError('Mixed reference five-benchmark comparison is inconsistent with its summary')
    mixed_dev = state['metrics']['svox_dev']
    if (_dev_identity(mixed_dev['definition']) != _dev_identity(dev_definition)
            or _metrics(mixed_dev) != comparison['svox_dev']['selected_percent']
            or not np.isclose(_metrics(mixed_dev)['R@1'], completed['best']['score'], atol=1e-10, rtol=0)):
        raise ValueError('Mixed reference SVOX development metrics/protocol do not match the new run')
    query_evidence = {}
    for name in DATASETS:
        row = summary['datasets'][name]
        identity = dict(schema=1, role='query_student', model_sha256=checkpoint_sha,
            image_size=config['image_size'], descriptor_dim=config['descriptor_dim'],
            transform=TRANSFORM, precision='float32', tf32=False, paths=row['queries'])
        query = root / 'final_cache' / name / 'query.npy'
        if (row['query_cache']['provenance'] != identity
                or Path(row['query_cache']['path']).resolve(strict=True) != query.resolve(strict=True)
                or row['gallery_cache']['feature_sha256'] != initial['datasets'][name]['gallery_cache']['feature_sha256']
                or row['gallery_cache']['provenance'] != initial['datasets'][name]['gallery_cache']['provenance']):
            raise ValueError('Mixed reference query/gallery checkpoint-space provenance mismatch')
        _check_query_cache(query, identity)
        query_sha = sha256(query)
        predicted = summary_path.parent / f'{name}_top10.npy'
        if row['query_cache']['feature_sha256'] != query_sha or row['predictions_sha256'] != sha256(predicted):
            raise ValueError('Mixed reference feature or top-10 checksum mismatch')
        query_evidence[name] = dict(feature_sha256=query_sha, predictions_sha256=sha256(predicted))
    evidence = dict(summary=str(summary_path), summary_sha256=sha256(summary_path),
        comparison=str(comparison_path), comparison_sha256=sha256(comparison_path),
        finalization=str(receipt_path), finalization_sha256=sha256(receipt_path),
        completed=str(completed_path), completed_sha256=sha256(completed_path),
        checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        svox_development_identity_sha256=json_sha256(_dev_identity(mixed_dev['definition'])),
        query_caches=query_evidence)
    del state
    return dict(summary=summary, comparison=comparison, identity=evidence,
                svox_dev_percent=_metrics(mixed_dev), tokyo_predictions=summary_path.parent / 'tokyo247_top10.npy')


def _receipt_identity_matches(prior, current, config):
    if not isinstance(prior, dict) or not isinstance(current, dict):
        return False
    if prior == current:
        return True
    # Previously completed v1 artifacts retain the exact old implementation
    # identity. Reading their intact sealed outputs does not rerun that code.
    if 'query_domain' in config or 'selection_policy' in config:
        return False
    old, new = json.loads(json.dumps(prior)), json.loads(json.dumps(current))
    old.get('implementation_sha256', {}).pop('night_finalize.py', None)
    new.get('implementation_sha256', {}).pop('night_finalize.py', None)
    return old == new


def compare_summaries(initial, final):
    """Paired five-benchmark comparison; refuse changes to gallery, queries or GT."""
    if set(initial['datasets']) != set(DATASETS) or set(final['datasets']) != set(DATASETS):
        raise ValueError('Fixed final comparison requires all five benchmarks')
    if initial.get('teacher_sha256') != final.get('teacher_sha256'):
        raise ValueError('Fixed final comparison changed the teacher gallery model')
    result = {}
    for name in DATASETS:
        before, after = initial['datasets'][name], final['datasets'][name]
        _assert_same_protocol(before, after, name)
        a, b = _metrics(before), _metrics(after)
        result[name] = dict(initial_percent=a, final_percent=b,
                           difference_percentage_points={k: b[k] - a[k] for k in a},
                           num_references=after['references']['count'], num_queries=after['queries']['count'])
    return result


def _validated_gallery(source, identity, initial_row):
    """Read-only source validation; never call a writable cache opener on originals."""
    source = Path(source)
    metadata = source.with_suffix('.provenance.json')
    progress_path = source.with_suffix('.progress.json')
    provenance, progress = json.loads(metadata.read_text()), json.loads(progress_path.read_text())
    expected_shape = (identity['paths']['count'], identity['descriptor_dim'])
    if provenance != identity or initial_row['gallery_cache']['provenance'] != identity:
        raise ValueError(f'Teacher gallery provenance mismatch: {source}')
    if (progress.get('identity_sha256') != json_sha256(identity)
            or progress.get('rows') != expected_shape[0] or progress.get('total') != expected_shape[0]):
        raise ValueError(f'Teacher gallery is incomplete or has another identity: {source}')
    values = np.load(source, mmap_mode='r', allow_pickle=False)
    if values.dtype != np.float32 or values.shape != expected_shape:
        raise ValueError(f'Teacher gallery shape/dtype mismatch: {source}')
    del values
    checksum = sha256(source)
    if (checksum != progress.get('feature_sha256')
            or checksum != initial_row['gallery_cache']['feature_sha256']):
        raise ValueError(f'Teacher gallery checksum mismatch: {source}')
    return dict(path=str(source.resolve()), feature_sha256=checksum,
                provenance_sha256=sha256(metadata), progress_sha256=sha256(progress_path),
                identity=identity)


def link_gallery_cache(source, destination, identity, initial_row):
    """Create only the three verified gallery symlinks; never link student queries."""
    source, destination = Path(source), Path(destination)
    evidence = _validated_gallery(source, identity, initial_row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.npy', '.provenance.json', '.progress.json'):
        origin, link = source.with_suffix(suffix), destination.with_suffix(suffix)
        if link.is_symlink():
            if link.resolve(strict=True) != origin.resolve(strict=True):
                raise ValueError(f'Final gallery symlink points to a different source: {link}')
        elif link.exists():
            raise ValueError(f'Refusing to overwrite a non-symlink final gallery cache: {link}')
        else:
            link.symlink_to(origin.resolve(strict=True))
    return evidence


def _check_query_cache(path, identity):
    """Queries may resume THIS best only; old-result symlinks are never accepted."""
    path = Path(path)
    companions = [path, path.with_suffix('.provenance.json'), path.with_suffix('.progress.json')]
    if any(p.is_symlink() for p in companions):
        raise ValueError('Final student query caches must be new regular files, never symlinks')
    present = [p.exists() for p in companions]
    if any(present) and not all(present):
        raise ValueError(f'Final query cache has incomplete sidecars: {path}')
    if not any(present):
        return
    provenance, progress = json.loads(companions[1].read_text()), json.loads(companions[2].read_text())
    if provenance != identity or progress.get('identity_sha256') != json_sha256(identity):
        raise ValueError(f'Final query cache belongs to another checkpoint/protocol: {path}')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (identity['paths']['count'], identity['descriptor_dim']):
        raise ValueError(f'Final query cache shape/dtype mismatch: {path}')
    count = len(values)
    del values
    done = progress.get('rows')
    if not isinstance(done, int) or not 0 <= done <= count or progress.get('total') != count:
        raise ValueError('Invalid final query-cache committed row count')
    if done == count and progress.get('feature_sha256') != sha256(path):
        raise ValueError('Completed final query cache checksum mismatch')


def _benchmarks(initial):
    roots = {name: initial['datasets'][name]['root'] for name in DATASETS}
    nordland = initial['datasets']['nordland']['protocol']['protocol_variant']
    return [build_benchmark(name, roots=roots, nordland_protocol=nordland) for name in DATASETS]


def _validate_final_summary(final, initial, best_sha, teacher_sha, cache, benches):
    if (final.get('checkpoint_sha256') != best_sha or final.get('teacher_sha256') != teacher_sha
            or final.get('precision') != 'float32' or final.get('tf32') is not False):
        raise ValueError('Final summary is bound to different weights, teacher or precision')
    compare_summaries(initial, final)
    for benchmark in benches:
        name, row = benchmark.name, final['datasets'][benchmark.name]
        if row['query_cache']['provenance']['model_sha256'] != best_sha:
            raise ValueError('Final result reused a query embedding from another checkpoint')
        query = cache / name / 'query.npy'
        if Path(row['query_cache']['path']).resolve() != query.resolve():
            raise ValueError('Final result query cache is outside the new final cache')
        if row['query_cache']['feature_sha256'] != sha256(query):
            raise ValueError('Final summary query-feature checksum mismatch')
        if row['gallery_cache']['feature_sha256'] != initial['datasets'][name]['gallery_cache']['feature_sha256']:
            raise ValueError('Final result did not reuse the unchanged teacher gallery')
        predicted = cache.parent / 'final_evaluation' / f'{name}_top10.npy'
        if sha256(predicted) != row['predictions_sha256']:
            raise ValueError('Final top-10 predictions differ from their summary checksum')


def tokyo_night_diagnostic(benchmark, initial_predictions, final_predictions,
                            source_path=TOKYO_SOURCE, mat_path=TOKYO_MAT):
    """Derive N-labeled queries from existing top-10; no additional model inference.

    The trusted local source binds the official NetVLAD MAT SHA. CSV field 6
    supplies D/S/N labels, while official MAT coordinates define 25 m positives.
    """
    from scipy.io import loadmat
    from scipy.spatial import cKDTree
    source_path, mat_path = Path(source_path), Path(mat_path)
    if not source_path.is_file() or not mat_path.is_file():
        return dict(available=False, reason='Trusted Tokyo time-of-day source/MAT not present; full 315 is mixed D/S/N')
    source = json.loads(source_path.read_text())
    if (source.get('dataset') != 'Tokyo247' or source.get('split') != 'test'
            or sha256(mat_path) != source.get('mat_sha256')
            or source.get('night_label_rule') != 'queries/<qImageFn stem>.csv, column 6 (zero-based index 5) == N'):
        raise ValueError('Trusted Tokyo night source/MAT provenance mismatch')
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)['dbStruct']
    if data.whichSet != 'test' or float(data.posDistThr) != 25. or float(data.posDistSqThr) != 625.:
        raise ValueError('Tokyo night diagnostic requires official test MAT and 25 m positives')
    root = benchmark.root
    official_refs = [(root / 'database_gsv_vga' / Path(str(n)).with_suffix('.png')).resolve()
                     for n in np.atleast_1d(data.dbImageFns)]
    official_queries = [str(n) for n in np.atleast_1d(data.qImageFns)]
    if (len(set(official_refs)) != len(official_refs) or len(set(official_queries)) != len(official_queries)
            or len(official_refs) != source['gallery_count'] or len(official_queries) != source['official_full_query_count']
            or set(official_refs) != set(benchmark.references)
            or {p.name for p in benchmark.queries} != set(official_queries)):
        raise ValueError('Tokyo complete query/gallery identities differ from bound official MAT')
    current_ref_index = {path: i for i, path in enumerate(benchmark.references)}
    official_to_current = np.asarray([current_ref_index[path] for path in official_refs], np.int64)
    query_index = {name: i for i, name in enumerate(official_queries)}
    selected, night_ids, csv_hashes = [], [], []
    for index, query in enumerate(benchmark.queries):
        metadata = query.with_suffix('.csv')
        with metadata.open(newline='') as stream:
            rows = list(csv.reader(stream))
        if (len(rows) != 1 or len(rows[0]) < 9 or rows[0][0] != query.name
                or rows[0][5] not in ('D', 'S', 'N')):
            raise ValueError(f'Invalid Tokyo time-of-day metadata: {metadata}')
        csv_hashes.append([query.name, sha256(metadata)])
        if rows[0][5] == 'N':
            selected.append(index); night_ids.append(query.name)
    if len(selected) != source['night_query_count'] or set(night_ids) != set(source['night_ids']):
        raise ValueError('Tokyo night IDs do not match the independently bound N-label metadata')
    ref_xy, query_xy = np.asarray(data.utmDb, np.float64).T, np.asarray(data.utmQ, np.float64).T
    if (ref_xy.shape != (len(official_refs), 2) or query_xy.shape != (len(official_queries), 2)
            or not np.isfinite(ref_xy).all() or not np.isfinite(query_xy).all()):
        raise ValueError('Invalid official Tokyo UTM coordinates')
    gt = cKDTree(ref_xy).query_ball_point(query_xy[[query_index[n] for n in night_ids]], 25.)
    positives = [np.sort(official_to_current[row]) for row in gt]
    if any(not len(row) for row in positives):
        raise ValueError('Tokyo night query has no official positive; refusing silent exclusion')
    before = np.load(initial_predictions, allow_pickle=False)
    after = np.load(final_predictions, allow_pickle=False)
    if before.shape != (len(benchmark.queries), 10) or after.shape != before.shape:
        raise ValueError('Tokyo top-10 rows do not match the full fixed query order')
    from reproduction.evaluate import recall_at_k
    a = recall_at_k(before[selected], positives)
    b = recall_at_k(after[selected], positives)
    before_correct = [np.isin(row[:1], gt).any() for row, gt in zip(before[selected], positives)]
    after_correct = [np.isin(row[:1], gt).any() for row, gt in zip(after[selected], positives)]
    return dict(available=True, num_queries=len(selected), num_references=len(benchmark.references),
                query_indices_in_full315=selected, query_ids=night_ids,
                initial_percent=a, final_percent=b,
                difference_percentage_points={k: b[k] - a[k] for k in a},
                improved_at_1=int(sum(not x and y for x, y in zip(before_correct, after_correct))),
                regressed_at_1=int(sum(x and not y for x, y in zip(before_correct, after_correct))),
                definition=dict(selection='CSV field 6 == N; verified against bound official 105-query ID set',
                    ground_truth='official MAT UTM <=25 m; current full-gallery indices mapped by exact image path',
                    selection_role='fixed final diagnostic, never checkpoint selection',
                    source_sha256=sha256(source_path), mat_sha256=sha256(mat_path),
                    csv_labels_sha256=json_sha256(csv_hashes),
                    ground_truth_sha256=json_sha256([p.tolist() for p in positives]),
                    differing_gt_rows_from_main_protocol=sum(not np.array_equal(p, benchmark.positives[i])
                                                            for p, i in zip(positives, selected)),
                    limitation=source.get('local_query_release_note', '')))


def _report(config, completed, state, baseline, comparison, tokyo, identity, mixed=None):
    before = _metrics(baseline['svox_dev'])
    after = _metrics(state['metrics']['svox_dev'])
    selection = completed['best']
    _validate_mixed_domain(config)
    policy = f"选模依据为 SVOX 真实夜间开发集 R@1，MSLS R@1 相对初始下降不超过 {config['max_msls_drop_pp']:g} 个百分点。"
    budget = ''
    lines = ['# AsymVPR 夜间继续训练报告', '',
        f"选中 epoch {state['epoch']}；adapted={selection['adapted']}。{policy}{budget}", '',
        f"SVOX 开发集 R@1：{before['R@1']:.3f}% → {after['R@1']:.3f}%（{after['R@1']-before['R@1']:+.3f} pp）。它是同城、同相机的目标域开发集，不是未见域测试。", '',
        '| 固定数据集 | 初始 R@1 (%) | 最终 R@1 (%) | Δ (pp) | 初始 R@5 (%) | 最终 R@5 (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for name, row in comparison.items():
        a, b = row['initial_percent'], row['final_percent']
        lines.append(f"| {name} | {a['R@1']:.3f} | {b['R@1']:.3f} | {row['difference_percentage_points']['R@1']:+.3f} | {a['R@5']:.3f} | {b['R@5']:.3f} |")
    lines += ['', 'Tokyo24/7 主表为完整 315 张混合时段查询。']
    if tokyo['available']:
        lines += [f"可靠 CSV 时间标签识别出 {tokyo['num_queries']} 张夜间查询。复用完整图库与既有 top-10，按官方 MAT 坐标派生的夜间 R@1：{tokyo['initial_percent']['R@1']:.3f}% → {tokyo['final_percent']['R@1']:.3f}%（{tokyo['difference_percentage_points']['R@1']:+.3f} pp）；修好 {tokyo['improved_at_1']}，改坏 {tokyo['regressed_at_1']}。这项固定最终诊断没有参与选模。"]
    if mixed is not None:
        a, b = mixed['svox_dev']['mixed_percent'], mixed['svox_dev']['selected_percent']
        lines += ['', f"相对 mixed best：SVOX 夜间开发集 R@1 {a['R@1']:.3f}% → {b['R@1']:.3f}%（{b['R@1']-a['R@1']:+.3f} pp）。", '',
                  '| 相对 mixed best | mixed R@1 (%) | 本实验 R@1 (%) | Δ (pp) |',
                  '|---|---:|---:|---:|']
        for name, row in mixed['five_benchmark_comparison'].items():
            lines.append(f"| {name} | {row['initial_percent']['R@1']:.3f} | {row['final_percent']['R@1']:.3f} | {row['difference_percentage_points']['R@1']:+.3f} |")
        diagnostic = mixed['tokyo_night']
        if diagnostic['available']:
            lines += ['', f"相对 mixed best 的 Tokyo105 夜间 R@1：{diagnostic['initial_percent']['R@1']:.3f}% → {diagnostic['final_percent']['R@1']:.3f}%（{diagnostic['difference_percentage_points']['R@1']:+.3f} pp）。"]
        lines += ['', '原始模型、mixed 和本实验使用同一教师空间、输入列表及真值。SVOX 开发集已被多个实验用于选模；Tokyo 夜间结果是重复使用的固定最终诊断，不作为全新独立测试声称。']
    supervision_note = '冻结教师日夜一致性权重也不是质量通过证明。'
    lines += ['', '生成图保留原始 pending_verification 状态。来源 SHA、完整解码、配对和去重检查不构成生成内容的语义审核；' + supervision_note + '合成验证只留出了新增夜间增强，初始模型已见过其白天地点。', '',
        '所有固定测试使用同一 BoQ 教师图库；只重新提取选中学生的查询。评测为 FP32、禁用 TF32、精确内积检索；Nordland、Tokyo 等协议与初始评测保持一致。', '',
        f"Best SHA256：`{identity['best_sha256']}`。冻结夜图清单 SHA256：`{identity['snapshot_manifest_sha256']}`。", '',
        '训练配置：', '', '```json', json.dumps(config, ensure_ascii=False, indent=2), '```', '']
    return '\n'.join(lines)


def finalize(config):
    """Validate completed training, evaluate selected best once, and seal a report."""
    train = Path(config['output_dir']).resolve()
    _validate_mixed_domain(config)
    root, cache, output = train.parent, train.parent / 'final_cache', train.parent / 'final_evaluation'
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.finalize.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Final night evaluation is already running for this experiment') from None
        completed_path, best_path, baseline_path = train / 'completed.json', train / 'best.pt', train / 'baseline.json'
        completed, baseline = json.loads(completed_path.read_text()), json.loads(baseline_path.read_text())
        if (completed.get('status') != 'completed' or completed.get('epochs') != config['epochs']
                or completed.get('config') != config):
            raise ValueError('Final evaluation requires this exact completed training configuration')
        teacher_sha, best_sha = sha256(config['teacher_weights']), sha256(best_path)
        state = load_inference_checkpoint(best_path)
        validate_inference_checkpoint(state, config, teacher_sha)
        if state.get('config') != config or any(state.get(k) != completed['best'].get(k) for k in ('epoch', 'step')):
            raise ValueError('Selected best checkpoint does not match completed experiment selection')
        if completed['best'].get('adapted') != (state['epoch'] > 0):
            raise ValueError('Completed adaptation flag differs from selected checkpoint epoch')
        if not np.isclose(_metrics(state['metrics']['svox_dev'])['R@1'], completed['best']['score'], atol=1e-10, rtol=0):
            raise ValueError('Selected best development score differs from completed selection')
        initial = json.loads(SOURCE_SUMMARY.read_text())
        if (initial.get('checkpoint_sha256') != sha256(config['init_checkpoint'])
                or initial.get('teacher_sha256') != teacher_sha or set(initial['datasets']) != set(DATASETS)):
            raise ValueError('Initial reference evaluation belongs to different weights, teacher or benchmark inventory')
        initial_predictions = {}
        for name in DATASETS:
            actual = sha256(SOURCE_SUMMARY.parent / f'{name}_top10.npy')
            if actual != initial['datasets'][name].get('predictions_sha256'):
                raise ValueError(f'Initial top-10 predictions differ from their summary checksum: {name}')
            initial_predictions[name] = actual
        manifest = Path(config['input_dir']) / 'records.jsonl'
        snapshot_summary = Path(config['input_dir']) / 'summary.json'
        snapshot = json.loads(snapshot_summary.read_text())
        if snapshot.get('manifest_sha256') != sha256(manifest):
            raise ValueError('Frozen night snapshot manifest no longer matches its summary receipt')
        dev_definition = baseline['svox_dev']['definition']
        if (dev_definition != state['metrics']['svox_dev']['definition']
                or dev_definition.get('manifest_sha256') != sha256(config['svox_dev_manifest'])
                or dev_definition.get('protocol_receipt_sha256') != sha256(Path(config['svox_dev_manifest']).parent / 'protocol.json')):
            raise ValueError('Selected and baseline SVOX development protocols/metadata differ')
        mixed = _mixed_reference(config, initial, dev_definition)
        receipt = root / 'finalization.json'
        identity = dict(schema=1, best_sha256=best_sha, teacher_sha256=teacher_sha,
            config_sha256=json_sha256(config), snapshot_manifest_sha256=sha256(manifest),
            snapshot_summary_sha256=sha256(snapshot_summary),
            svox_development_manifest_sha256=sha256(config['svox_dev_manifest']),
            svox_development_protocol_sha256=sha256(Path(config['svox_dev_manifest']).parent / 'protocol.json'),
            training_completed_sha256=sha256(completed_path), baseline_sha256=sha256(baseline_path),
            initial_evaluation_sha256=sha256(SOURCE_SUMMARY),
            initial_predictions_sha256=initial_predictions,
            tokyo_time_source_sha256=sha256(TOKYO_SOURCE) if TOKYO_SOURCE.is_file() else None,
            tokyo_official_mat_sha256=sha256(TOKYO_MAT) if TOKYO_MAT.is_file() else None,
            source_gallery_state={name: dict(
                size=(SOURCE_CACHE / name / 'gallery.npy').stat().st_size,
                mtime_ns=(SOURCE_CACHE / name / 'gallery.npy').stat().st_mtime_ns,
                provenance_sha256=sha256(SOURCE_CACHE / name / 'gallery.provenance.json'),
                progress_sha256=sha256(SOURCE_CACHE / name / 'gallery.progress.json')) for name in DATASETS},
            implementation_sha256={p.name: sha256(p) for p in [Path(__file__), PROJECT / 'reproduction/evaluate.py',
                PROJECT / 'reproduction/benchmarks.py', PROJECT / 'reproduction/path_resolver.py']})
        if mixed is not None:
            identity['mixed_reference'] = mixed['identity']
        if receipt.is_file():
            prior = _read_sealed(receipt)
            if (not _receipt_identity_matches(prior.get('identity'), identity, config)
                    or prior.get('status') != 'complete'
                    or prior.get('identity_sha256') != json_sha256(prior.get('identity'))):
                raise ValueError('Finalization inputs changed; refusing to reuse or overwrite another final result')
            for path, checksum in prior['outputs_sha256'].items():
                if sha256(path) != checksum:
                    raise ValueError(f'Finalization output checksum changed: {path}')
            for name in DATASETS:
                for suffix in ('.npy', '.provenance.json', '.progress.json'):
                    link = (cache / name / 'gallery.npy').with_suffix(suffix)
                    original = (SOURCE_CACHE / name / 'gallery.npy').with_suffix(suffix)
                    if not link.is_symlink() or link.resolve(strict=True) != original.resolve(strict=True):
                        raise ValueError(f'Final teacher gallery reuse link changed: {link}')
            return prior
        benches = _benchmarks(initial)
        gallery_evidence = {}
        for benchmark in benches:
            name = benchmark.name
            _assert_same_protocol(initial['datasets'][name], benchmark.describe(), name)
            gallery_identity = feature_identity(benchmark.references, config, teacher_sha, 'gallery_teacher')
            gallery_evidence[name] = link_gallery_cache(SOURCE_CACHE / name / 'gallery.npy',
                cache / name / 'gallery.npy', gallery_identity, initial['datasets'][name])
            query_identity = feature_identity(benchmark.queries, config, best_sha, 'query_student')
            _check_query_cache(cache / name / 'query.npy', query_identity)
        output.mkdir(parents=True, exist_ok=True)
        eval_config, roots_path = output / 'config.json', output / 'roots.json'
        atomic_json(eval_config, config)
        atomic_json(roots_path, {b.name: str(b.root) for b in benches})
        summary_path = output / 'summary.json'
        existing = json.loads(summary_path.read_text()) if summary_path.is_file() else None
        if existing is not None and existing.get('checkpoint_sha256') != best_sha:
            raise ValueError('Existing final evaluation summary belongs to another best checkpoint')
        if existing is None or set(existing.get('datasets', {})) != set(DATASETS):
            command = [sys.executable, '-m', 'reproduction.evaluate', '--config', str(eval_config),
                '--checkpoint', str(best_path), '--dataset', 'all', '--cache-dir', str(cache),
                '--output-dir', str(output), '--roots-json', str(roots_path), '--nordland-protocol',
                initial['datasets']['nordland']['protocol']['protocol_variant'],
                '--batch-size', '128', '--workers', '4']
            environment = os.environ.copy()
            environment['TRAIN_TERMINAL_RUN_DIR'] = str(output / 'logs')
            print('Fixed final evaluation:', command, flush=True)
            # Inherit stdout/stderr: the enclosing Trainroom job keeps the real process logs.
            subprocess.run(command, cwd=PROJECT, env=environment, check=True)
        final = json.loads(summary_path.read_text())
        _validate_final_summary(final, initial, best_sha, teacher_sha, cache, benches)
        if sha256(best_path) != best_sha or json_sha256(config) != identity['config_sha256']:
            raise ValueError('Best weights or training configuration changed during final evaluation')
        comparison = compare_summaries(initial, final)
        tokyo = tokyo_night_diagnostic(next(b for b in benches if b.name == 'tokyo247'),
            SOURCE_SUMMARY.parent / 'tokyo247_top10.npy', output / 'tokyo247_top10.npy')
        mixed_comparison = None
        if mixed is not None:
            for path_key, sha_key in [('summary', 'summary_sha256'), ('comparison', 'comparison_sha256'),
                                     ('finalization', 'finalization_sha256'), ('checkpoint', 'checkpoint_sha256')]:
                if sha256(mixed['identity'][path_key]) != mixed['identity'][sha_key]:
                    raise ValueError('Mixed reference changed during final evaluation')
            night = tokyo_night_diagnostic(next(b for b in benches if b.name == 'tokyo247'),
                mixed['tokyo_predictions'], output / 'tokyo247_top10.npy')
            recorded_night = mixed['comparison']['tokyo_night']
            if night['available'] != recorded_night['available']:
                raise ValueError('Mixed Tokyo night diagnostic uses another metadata availability protocol')
            if night['available'] and any(night[key] != recorded_night[old_key] for key, old_key in
                [('initial_percent', 'final_percent'), ('definition', 'definition'),
                 ('query_ids', 'query_ids'), ('query_indices_in_full315', 'query_indices_in_full315')]):
                raise ValueError('Mixed Tokyo night metrics/definition differ from bound top-10 predictions')
            a, b = mixed['svox_dev_percent'], _metrics(state['metrics']['svox_dev'])
            mixed_comparison = dict(reference=mixed['identity'],
                five_benchmark_comparison=compare_summaries(mixed['summary'], final),
                svox_dev=dict(mixed_percent=a, selected_percent=b,
                              difference_percentage_points={k: b[k]-a[k] for k in a}),
                tokyo_night=night)
        comparison_path = output / 'comparison.json'
        details = dict(adapted=completed['best']['adapted'], selected=completed['best'],
            svox_dev=dict(initial_percent=_metrics(baseline['svox_dev']),
                          selected_percent=_metrics(state['metrics']['svox_dev'])),
            five_benchmark_comparison=comparison, tokyo_night=tokyo, training_config=config,
            limitations=['Generated night images remain pending semantic verification',
                         'SVOX train-night is same-city/camera development and selected the checkpoint',
                         'Synthetic holdout is new-augmentation holdout; original day places were pretrained'])
        if mixed_comparison is not None:
            details['mixed_reference_comparison'] = mixed_comparison
        atomic_json(comparison_path, _sealed(details))
        report = root / 'NIGHT_REPORT.md'
        report.write_text(_report(config, completed, state, baseline, comparison, tokyo, identity, mixed_comparison))
        outputs = [summary_path, comparison_path, report, eval_config, roots_path]
        outputs += [output / f'{name}_top10.npy' for name in DATASETS]
        result = dict(schema=1, status='complete', identity=identity,
            identity_sha256=json_sha256(identity), comparison=details, gallery_reuse=gallery_evidence,
            outputs_sha256={str(p.resolve()): sha256(p) for p in outputs})
        atomic_json(receipt, _sealed(result))
        return result
