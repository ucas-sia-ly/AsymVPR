import json
import pytest
from reproduction import mixed_reference as reference
from reproduction.benchmarks import json_sha256

def test_reference_preflight_protects_original_run_and_controlled_budget(tmp_path):
    old = tmp_path / 'mixed/train'; old.mkdir(parents=True)
    (old / 'best.pt').write_bytes(b'selected mixed weights')
    config = dict(init_checkpoint='base.pt', input_dir='fixed_snapshot', night_teacher_cache='fixed.npy',
        base_cache='fixed_bank', teacher_weights='teacher.pth', svox_dev_manifest='nightdev.json',
        seed=2026, night_batch_size=32, epochs=10, lr_backbone=1e-5, lr_aggregator=5e-5,
        min_lr_ratio=.2, temperature=.05, gamma=15., output_dir=str(old))
    comparison = {'training_config': dict(config)}
    comparison['record_sha256'] = json_sha256(comparison)
    cp, sm = tmp_path / 'comparison.json', tmp_path / 'summary.json'
    cp.write_text(json.dumps(comparison))
    sm.write_text(json.dumps({'checkpoint_sha256': reference.sha256(old / 'best.pt')}))
    config.update(output_dir=str(tmp_path / 'camera/train'),
        mixed_reference_comparison=str(cp), mixed_reference_summary=str(sm),
        mixed_reference_comparison_sha256=reference.sha256(cp), mixed_reference_summary_sha256=reference.sha256(sm))
    assert reference.validate_reference_inputs(config)['checkpoint_sha256'] == reference.sha256(old / 'best.pt')
    with pytest.raises(ValueError, match='outside the original'):
        reference.validate_reference_inputs(dict(config, output_dir=str(old)))
    with pytest.raises(ValueError, match='differs from mixed reference: epochs'):
        reference.validate_reference_inputs(dict(config, epochs=20))
    with pytest.raises(ValueError, match='checksum mismatch'):
        reference.validate_reference_inputs(dict(config, mixed_reference_summary_sha256='incorrect'))
