"""Sealed camera-on/off comparison after the normal fixed final evaluation."""
import fcntl
import json
from pathlib import Path

from reproduction.benchmarks import DATASETS, json_sha256
from reproduction.evaluate import atomic_json, load_inference_checkpoint, sha256, validate_inference_checkpoint
from reproduction.night_finalize import (_benchmarks, _dev_identity, _metrics, _read_sealed,
    _sealed, compare_summaries, tokyo_night_diagnostic)


def validate_night_reference(config):
    paths = {}
    for kind in ('summary', 'comparison'):
        key = f'night_only_reference_{kind}'
        path = Path(config[key]).resolve(strict=True)
        if sha256(path) != config.get(key + '_sha256'):
            raise ValueError(f'Camera-off reference {kind} SHA256 mismatch')
        paths[kind] = path
    if paths['summary'].parent != paths['comparison'].parent:
        raise ValueError('Camera-off summary and comparison must belong to one experiment')
    root = paths['summary'].parent.parent
    completed_path, receipt_path = root / 'train/completed.json', root / 'finalization.json'
    receipt, comparison = _read_sealed(receipt_path), _read_sealed(paths['comparison'])
    summary = json.loads(paths['summary'].read_text())
    completed = json.loads(completed_path.read_text())
    if (receipt.get('status') != 'complete' or completed.get('status') != 'completed'
            or receipt.get('identity_sha256') != json_sha256(receipt['identity'])
            or receipt.get('comparison') != comparison
            or receipt['identity']['training_completed_sha256'] != sha256(completed_path)):
        raise ValueError('Camera-off reference is not the bound completed experiment')
    for path in paths.values():
        if receipt['outputs_sha256'].get(str(path)) != sha256(path):
            raise ValueError('Camera-off reference output differs from its sealed receipt')
    checkpoint = root / 'train/best.pt'
    checksum, teacher_sha = sha256(checkpoint), sha256(config['teacher_weights'])
    state = load_inference_checkpoint(checkpoint)
    validate_inference_checkpoint(state, config, teacher_sha)
    old = state['config']
    if (old.get('camera_profile') or old.get('query_domain') != 'night_only'
            or completed['config'] != old or comparison['training_config'] != old
            or receipt['identity']['config_sha256'] != json_sha256(old)
            or receipt['identity']['best_sha256'] != checksum
            or summary['checkpoint_sha256'] != checksum or summary['teacher_sha256'] != teacher_sha
            or summary.get('precision') != 'float32' or summary.get('tf32') is not False
            or Path(summary['checkpoint']).resolve() != checkpoint.resolve()
            or any(state[k] != completed['best'][k] for k in ('epoch', 'step'))
            or completed['best'] != comparison['selected']):
        raise ValueError('Camera-off selected checkpoint/config/provenance mismatch')
    controls = ('init_checkpoint', 'input_dir', 'night_teacher_cache', 'base_cache', 'teacher_weights',
        'image_size', 'student_name', 'descriptor_dim', 'proj_channels', 'num_queries', 'num_layers',
        'svox_dev_manifest', 'seed', 'night_batch_size', 'replay_batch_size', 'epochs', 'total_optimizer_steps',
        'lr_backbone', 'lr_aggregator', 'min_lr_ratio', 'weight_decay', 'temperature', 'gamma',
        'precision', 'night_target', 'selection_policy', 'max_msls_drop_pp')
    for key in controls:
        if old[key] != config[key]:
            raise ValueError(f'Camera on/off experiment changes a controlled field: {key}')
    output = Path(config['output_dir']).resolve()
    if output == root or root in output.parents:
        raise ValueError('Camera-on outputs must be outside the camera-off experiment')
    dev = state['metrics']['svox_dev']
    if (_metrics(dev) != comparison['svox_dev']['selected_percent']
            or _metrics(dev)['R@1'] != completed['best']['score']
            or dev['definition']['manifest_sha256'] != sha256(config['svox_dev_manifest'])):
        raise ValueError('Camera-off SVOX development metrics/manifest mismatch')
    predictions = {}
    for name in DATASETS:
        path = paths['summary'].parent / f'{name}_top10.npy'
        checksum_pred = sha256(path)
        if (summary['datasets'][name]['predictions_sha256'] != checksum_pred
                or receipt['outputs_sha256'].get(str(path)) != checksum_pred):
            raise ValueError(f'Camera-off prediction checksum mismatch: {name}')
        predictions[name] = checksum_pred
    identity = dict(summary_sha256=sha256(paths['summary']), comparison_sha256=sha256(paths['comparison']),
        finalization_sha256=sha256(receipt_path), checkpoint_sha256=checksum,
        completed_sha256=sha256(completed_path), predictions_sha256=predictions)
    return dict(identity=identity, summary=summary, comparison=comparison, selected=completed['best'],
                dev=dev, summary_path=str(paths['summary']))


def finalize_camera(config):
    """Read completed clean-query tests, compare camera off/on, never run extra tests."""
    from reproduction.night_camera import camera_identity
    provenance = camera_identity(config)
    train = Path(config['output_dir']).resolve()
    root, output = train.parent, train.parent / 'final_evaluation'
    with (root / '.camera_finalize.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        reference = validate_night_reference(config)
        completed = json.loads((train / 'completed.json').read_text())
        receipt = _read_sealed(root / 'finalization.json')
        final_path = output / 'summary.json'
        final = json.loads(final_path.read_text())
        current_comparison = _read_sealed(output / 'comparison.json')
        if (completed.get('status') != 'completed' or completed['config'] != config
                or receipt.get('status') != 'complete' or receipt['comparison'] != current_comparison
                or receipt.get('identity_sha256') != json_sha256(receipt['identity'])
                or receipt['identity']['config_sha256'] != json_sha256(config)
                or receipt['identity']['training_completed_sha256'] != sha256(train / 'completed.json')
                or receipt['outputs_sha256'].get(str(final_path)) != sha256(final_path)
                or receipt['identity']['best_sha256'] != sha256(train / 'best.pt')
                or final['checkpoint_sha256'] != receipt['identity']['best_sha256']):
            raise ValueError('Camera comparison requires the bound completed fixed final evaluation')
        state = load_inference_checkpoint(train / 'best.pt')
        if (state.get('config') != config or current_comparison['selected'] != completed['best']
                or _metrics(state['metrics']['svox_dev']) != current_comparison['svox_dev']['selected_percent']
                or _dev_identity(reference['dev']['definition']) != _dev_identity(state['metrics']['svox_dev']['definition'])
                or any(state[k] != completed['best'][k] for k in ('epoch', 'step'))):
            raise ValueError('Camera on/off development definitions or selected counters differ')
        five = compare_summaries(reference['summary'], final)
        for name in DATASETS:
            before = reference['summary']['datasets'][name]['gallery_cache']
            after = final['datasets'][name]['gallery_cache']
            if before['feature_sha256'] != after['feature_sha256'] or before['provenance'] != after['provenance']:
                raise ValueError('Camera on/off teacher gallery changed')
            path = output / f'{name}_top10.npy'
            if sha256(path) != final['datasets'][name]['predictions_sha256']:
                raise ValueError('Camera-on predictions differ from the final summary')
        identity = dict(schema=1, reference=reference['identity'],
            finalization_sha256=sha256(root / 'finalization.json'), config_sha256=json_sha256(config),
            final_summary_sha256=sha256(final_path), camera=provenance,
            implementation_sha256=sha256(__file__))
        comparison_path, report_path = output / 'camera_comparison.json', root / 'CAMERA_REPORT.md'
        marker = root / 'camera_finalization.json'
        if marker.exists():
            prior = _read_sealed(marker)
            if prior['identity'] != identity or any(sha256(p) != h for p, h in prior['outputs_sha256'].items()):
                raise ValueError('Camera finalization inputs or sealed outputs changed')
            return prior
        benchmark = next(b for b in _benchmarks(final) if b.name == 'tokyo247')
        night = tokyo_night_diagnostic(benchmark,
            Path(reference['summary_path']).parent / 'tokyo247_top10.npy', output / 'tokyo247_top10.npy')
        old_night = reference['comparison']['tokyo_night']
        if night['available'] != old_night['available']:
            raise ValueError('Camera on/off Tokyo night availability differs')
        if night['available'] and any(night[key] != old_night[old_key] for key, old_key in
                [('initial_percent', 'final_percent'), ('definition', 'definition'), ('query_ids', 'query_ids')]):
            raise ValueError('Camera-off Tokyo night reference differs from its bound predictions')
        a, b = _metrics(reference['dev']), _metrics(state['metrics']['svox_dev'])
        details = dict(identity=identity, selected=completed['best'], camera_off_selected=reference['selected'],
            svox_dev=dict(camera_off_percent=a, camera_on_percent=b,
                difference_percentage_points={k: b[k]-a[k] for k in a}),
            five_benchmark_comparison=five, tokyo_night=night)
        lines = ['# 夜间退化 on/off 训练测试对照', '',
            '处理顺序：原尺寸 RGB 夜图 → 随机相机退化 → ToTensor → Resize → ImageNet Normalize。', '',
            '只对学生训练夜图加退化；教师目标使用对应未退化夜图。真实验证和固定最终测试均不加退化。', '',
            f"相同初始化、29,050 张训练夜图、10 epochs / 9,080 updates。选中 epoch {state['epoch']}，只按 SVOX night dev R@1 选模。", '',
            f"SVOX night dev R@1：无退化 {a['R@1']:.4f}% → 有退化 {b['R@1']:.4f}%（{b['R@1']-a['R@1']:+.4f} pp）。", '',
            '| 固定测试 R@1 (%) | 无退化 | 有退化 | Δ (pp) |', '|---|---:|---:|---:|']
        for name, row in five.items():
            lines.append(f"| {name} | {row['initial_percent']['R@1']:.4f} | {row['final_percent']['R@1']:.4f} | {row['difference_percentage_points']['R@1']:+.4f} |")
        if night['available']:
            lines += ['', f"Tokyo105 夜间 R@1：{night['initial_percent']['R@1']:.4f}% → {night['final_percent']['R@1']:.4f}%（{night['difference_percentage_points']['R@1']:+.4f} pp）。"]
        lines += ['', 'SVOX 是参与选模的同域开发集；Tokyo105 是重复使用的固定最终诊断。单次种子结果不构成统计显著性结论。',
            '相机参数沿用 WORKSPACE exploratory profile，未用测试图校准。生成图仍为 pending_verification。', '',
            '详细配置与原始模型、mixed 对照见 NIGHT_REPORT.md；本报告专门对照无退化 night-only 模型。', '',
            f"Profile SHA256：`{provenance['profile_sha256']}`。", '']
        atomic_json(comparison_path, _sealed(details))
        report_path.write_text('\n'.join(lines))
        result = dict(status='complete', identity=identity, comparison=details,
            outputs_sha256={str(p): sha256(p) for p in (comparison_path, report_path)})
        atomic_json(marker, _sealed(result))
        return result
