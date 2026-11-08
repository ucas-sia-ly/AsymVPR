"""Pinned mixed baseline checks shared by the retained camera experiment."""
import json
from pathlib import Path
from reproduction.evaluate import sha256


def validate_reference_inputs(config):
    """Fail before GPU work if the fixed mixed reference or shared inputs differ."""
    from reproduction.night_finalize import _read_sealed
    summary_path = Path(config['mixed_reference_summary'])
    comparison_path = Path(config['mixed_reference_comparison'])
    for kind, path in [('summary', summary_path), ('comparison', comparison_path)]:
        if sha256(path) != config[f'mixed_reference_{kind}_sha256']:
            raise ValueError(f'Pinned mixed {kind} checksum mismatch')
    summary = json.loads(summary_path.read_text())
    comparison = _read_sealed(comparison_path)
    mixed_config = comparison['training_config']
    for key in ('init_checkpoint', 'input_dir', 'night_teacher_cache', 'base_cache', 'teacher_weights',
                'svox_dev_manifest', 'seed', 'night_batch_size', 'epochs', 'lr_backbone',
                'lr_aggregator', 'min_lr_ratio', 'temperature', 'gamma'):
        if mixed_config[key] != config[key]:
            raise ValueError(f'Controlled mixed-camera experiment differs from mixed reference: {key}')
    old_root = Path(mixed_config['output_dir']).resolve().parent
    new_output = Path(config['output_dir']).resolve()
    if new_output == old_root or old_root in new_output.parents:
        raise ValueError('Mixed-camera outputs must be outside the original mixed experiment')
    mixed_best = Path(mixed_config['output_dir']) / 'best.pt'
    if summary['checkpoint_sha256'] != sha256(mixed_best):
        raise ValueError('Mixed comparison no longer points to its actual selected checkpoint')
    return dict(summary_sha256=config['mixed_reference_summary_sha256'],
                comparison_sha256=config['mixed_reference_comparison_sha256'],
                checkpoint_sha256=summary['checkpoint_sha256'])
