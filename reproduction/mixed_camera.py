"""Controlled mixed-query camera ablation and a sealed direct comparison."""
import json
from pathlib import Path

from reproduction.benchmarks import json_sha256
from reproduction.evaluate import atomic_json, sha256
from reproduction.night_camera import camera_identity
from reproduction.night_finalize import _read_sealed, _sealed


def validate_config(config):
    from reproduction.night_only_train import validate_reference_inputs
    if (config.get('query_domain') != 'mixed'
            or not config.get('camera_profile')
            or config.get('selection_policy') != 'svox_night_r1_with_msls_guard'
            or config.get('night_target') != 'teacher_daynight_mix'
            or config.get('camera_student_branch') != 'night_only'
            or config.get('batch_size') != config.get('night_batch_size', 0) + config.get('replay_batch_size', 0)
            or config.get('precision') != 'float32'):
        raise ValueError('Mixed-camera requires mixed queries, the original fusion/day guard and night-only perturbations')
    validate_reference_inputs(config)
    comparison = _read_sealed(config['mixed_reference_comparison'])
    original = comparison['training_config']
    controls = ('batch_size', 'night_batch_size', 'replay_batch_size', 'gate_cos_min', 'gate_cos_span',
        'max_night_weight', 'max_msls_drop_pp', 'image_size', 'student_name', 'descriptor_dim',
        'proj_channels', 'num_queries', 'num_layers', 'weight_decay', 'bank_chunk_size', 'precision',
        'eval_cache_dir', 'msls_gallery_cache')
    for key in controls:
        if original[key] != config[key]:
            raise ValueError(f'Mixed-camera changes a controlled field: {key}')
    if config.get('total_optimizer_steps') != 9080:
        raise ValueError('Mixed-camera comparison requires the original 9080 optimizer updates')
    summary_path = Path(config['mixed_reference_summary']).resolve()
    receipt = _read_sealed(summary_path.parent.parent / 'finalization.json')
    if (receipt.get('status') != 'complete' or receipt['comparison'] != comparison
            or receipt.get('identity_sha256') != json_sha256(receipt['identity'])
            or receipt['outputs_sha256'].get(str(summary_path)) != sha256(summary_path)
            or receipt['outputs_sha256'].get(str(Path(config['mixed_reference_comparison']).resolve()))
                != sha256(config['mixed_reference_comparison'])):
        raise ValueError('Mixed camera-off reference does not match its completed sealed evaluation')
    if sha256(config['init_checkpoint']) != json.loads(
            (Path(original['output_dir']) / 'lineage.json').read_text())['init_sha256']:
        raise ValueError('Mixed camera-off/on initial checkpoint bytes differ')
    provenance = camera_identity(config)
    provenance.update(student_branches='degraded night + unchanged day replay in one forward',
        teacher_target='unchanged gated fusion of clean T(night) and clean T(source_day); clean T(replay_day)',
        selection='SVOX night dev R@1 with the original 1pp MSLS guard',
        mixed_reference_summary_sha256=config['mixed_reference_summary_sha256'],
        mixed_reference_comparison_sha256=config['mixed_reference_comparison_sha256'])
    return provenance


def report_completed(config, result):
    """No extra inference: the shared finalizer already verified mixed best and tests."""
    root = Path(config['output_dir']).resolve().parent
    receipt_path = root / 'finalization.json'
    body = _read_sealed(receipt_path)
    if body != result or body.get('status') != 'complete':
        raise ValueError('Mixed-camera report requires its intact fixed final evaluation receipt')
    for path, checksum in body['outputs_sha256'].items():
        if sha256(path) != checksum:
            raise ValueError('Mixed-camera final evaluation output checksum changed')
    provenance = validate_config(config)
    details = body['comparison']['mixed_reference_comparison']
    identity = dict(finalization_sha256=sha256(receipt_path), config_sha256=json_sha256(config),
        implementation_sha256=sha256(__file__), camera=provenance)
    marker = root / 'mixed_camera_finalization.json'
    if marker.exists():
        prior = _read_sealed(marker)
        if prior['identity'] != identity or any(sha256(p) != h for p, h in prior['outputs_sha256'].items()):
            raise ValueError('Mixed-camera report inputs or sealed outputs changed')
        return prior
    a, b = details['svox_dev']['mixed_percent'], details['svox_dev']['selected_percent']
    lines = ['# 混合训练＋夜间退化：直接对照', '',
        '每批 32 night + 32 day。只对学生夜图在原尺寸加七类退化，再执行 ToTensor → Resize → ImageNet Normalize。',
        '白天回放、日夜教师融合、损失权重及 MSLS 下降不超过 1pp 的选模门槛保持原样。教师和最终测试不附加人工退化。', '',
        f"同一起点、夜图划分和 10 epochs / 9,080 updates；本次选中 epoch {body['comparison']['selected']['epoch']}。", '',
        f"SVOX night dev R@1：混合无退化 {a['R@1']:.4f}% → 混合有退化 {b['R@1']:.4f}%（{b['R@1']-a['R@1']:+.4f} pp）。", '',
        '| 固定测试 R@1 (%) | 混合无退化 | 混合有退化 | Δ (pp) |', '|---|---:|---:|---:|']
    for name, row in details['five_benchmark_comparison'].items():
        lines.append(f"| {name} | {row['initial_percent']['R@1']:.4f} | {row['final_percent']['R@1']:.4f} | {row['difference_percentage_points']['R@1']:+.4f} |")
    night = details['tokyo_night']
    if night['available']:
        lines += ['', f"Tokyo105 night R@1：{night['initial_percent']['R@1']:.4f}% → {night['final_percent']['R@1']:.4f}%（{night['difference_percentage_points']['R@1']:+.4f} pp）。"]
    lines += ['', 'SVOX 是同域选模开发集；Tokyo105 是重复使用的固定最终诊断。单次种子不构成统计显著性结论。',
        'WORKSPACE 退化参数为 exploratory，生成图继续保留 pending_verification 状态。完整协议及原始模型对照见 NIGHT_REPORT.md。', '']
    report, data = root / 'MIXED_CAMERA_REPORT.md', root / 'final_evaluation/mixed_camera_comparison.json'
    report.write_text('\n'.join(lines))
    atomic_json(data, _sealed(dict(identity=identity, comparison=details)))
    sealed = dict(status='complete', identity=identity,
        outputs_sha256={str(p): sha256(p) for p in (report, data)})
    atomic_json(marker, _sealed(sealed))
    return sealed
