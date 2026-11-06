"""Exercise the actual specialist loop and complete-state resume on CPU."""
from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from reproduction import night_only_train as trainer
from reproduction.benchmarks import json_sha256


class TinyStudent(nn.Module):
    def __init__(self, config, pretrained=False):
        super().__init__()
        self.backbone = nn.Sequential(nn.Flatten(), nn.Linear(3*8*8, 8), nn.BatchNorm1d(8))
        self.aggregator = nn.Linear(8, 6)

    def forward(self, images):
        # Generated nights in this fixture have uniformly low normalized values.
        assert len(images) <= 3
        return F.normalize(self.aggregator(self.backbone(images)), dim=-1)


@pytest.fixture
def cpu_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(nn.Module, 'cuda', lambda self, *a, **kw: self)
    monkeypatch.setattr(torch.cuda, 'manual_seed_all', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(trainer, 'EfficientViTBoQ', TinyStudent)
    monkeypatch.setattr(trainer.shared, 'STOP', False)
    cache = tmp_path / 'cache'; cache.mkdir()
    gen = np.random.default_rng(1)
    for name, value in dict(train_features=gen.normal(size=(8, 6)).astype('float32'),
                            centroids=gen.normal(size=(4, 6)).astype('float32')*.1,
                            variances=np.full((4, 6), .01, np.float32)).items():
        np.save(cache / f'{name}.npy', value)
    night_teacher = gen.normal(size=(8, 6)).astype('float32')
    night_teacher /= np.linalg.norm(night_teacher, axis=1, keepdims=True)
    np.save(tmp_path / 'night.npy', night_teacher)
    records = []
    for i in range(8):
        path = tmp_path / f'night{i}.png'
        Image.new('RGB', (8, 8), (5+i, 10+i, 15+i)).save(path)
        records.append(dict(night_path=str(path), source_index=i, label=1 if i%2 else 3, split='train'))
    all_records = list(reversed(records))
    # Cache order must agree with the all-record index (not the train order).
    monkeypatch.setattr(trainer, 'load_records', lambda config, split: records)
    config = dict(base_cache=str(cache), night_teacher_cache=str(tmp_path / 'night.npy'),
        night_batch_size=3, replay_batch_size=0, epochs=2, seed=2026, image_size=8,
        lr_backbone=1e-4, lr_aggregator=5e-4, min_lr_ratio=.2, weight_decay=.01,
        temperature=.05, gamma=15., bank_chunk_size=2, num_workers=0, prefetch_factor=1,
        log_every=1, save_every=2)
    torch.manual_seed(55)
    initial = deepcopy(TinyStudent(config).state_dict())
    monkeypatch.setattr(trainer, 'load_inference_checkpoint', lambda path: {'model': deepcopy(initial)})
    config['init_checkpoint'] = 'unused-in-CPU-fixture'
    calls = []

    def validation(model, config, epoch, step, logger):
        calls.append((epoch, step))
        # Deliberately severe day degradation must not block night selection.
        return {'msls': {'metrics_percent': {'R@1': 92.-epoch*20}},
                'svox_dev': {'metrics_percent': {'R@1': 80.+step}}}

    monkeypatch.setattr(trainer.shared, 'validation', validation)
    return config, all_records, calls


class Logger:
    def __init__(self):
        self.rows = []

    def log(self, step, **values):
        self.rows.append(dict(step=step, **values))


def run(config, records, root, max_steps=None):
    config = dict(config, output_dir=str(root))
    root.mkdir(exist_ok=True)
    trainer.shared.seed_all(2026)
    logger = Logger()
    trainer.train(config, records, {'base_provenance': {}}, logger, max_steps)
    state = torch.load(root / 'last.pt', map_location='cpu', weights_only=False)
    return state, logger


def test_bounded_training_resumes_exactly_and_night_best_ignores_day_collapse(cpu_loop, tmp_path):
    config, records, calls = cpu_loop
    whole, whole_log = run(config, records, tmp_path / 'whole')
    calls.clear()
    first, first_log = run(config, records, tmp_path / 'resume', max_steps=2)
    assert (first['epoch'], first['next_batch'], first['step']) == (0, 2, 2)
    assert not (tmp_path / 'resume/completed.json').exists()
    resumed, second_log = run(config, records, tmp_path / 'resume')
    assert calls == [(0, 0), (1, 3), (2, 6)]  # Baseline not repeated on resume.
    assert (resumed['epoch'], resumed['next_batch'], resumed['step']) == (2, 0, 6)
    for key, value in whole['model'].items():
        torch.testing.assert_close(value, resumed['model'][key], rtol=0, atol=0)
    assert resumed['scheduler'] == whole['scheduler']
    assert resumed['best']['epoch'] == 2 and resumed['best']['score'] == 86.
    assert resumed['best']['msls_delta_pp'] == -40.
    assert resumed['best']['daytime_veto'] is False
    metrics = json.loads((tmp_path / 'resume/epoch_02_metrics.json').read_text())
    assert metrics['eligible_for_selection'] is True
    assert [r['step'] for r in first_log.rows if 'train/loss' in r] == [1, 2]
    assert [r['step'] for r in second_log.rows if 'train/loss' in r] == [3, 4, 5, 6]
    assert all('train/replay_asym' not in r for r in whole_log.rows)


def test_exposure_budget_matches_reference_updates_and_night_exposures():
    budget = trainer.exposure_budget({'night_batch_size': 32, 'epochs': 10}, 29050)
    assert budget['optimizer_steps'] == 9080
    assert budget['night_image_exposures'] == budget['total_student_image_exposures'] == 290500
    assert budget['day_query_image_exposures'] == 0


def test_msls_curves_are_explicitly_diagnostic():
    logger = Logger()
    trainer.DiagnosticValidationLogger(logger).log(908,
        **{'val/msls_guard/R@1': 45., 'val/svox_night_dev/R@1': 91.})
    assert logger.rows == [{'step': 908, 'diagnostic/msls/R@1': 45., 'val/svox_night_dev/R@1': 91.}]


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
    sm.write_text(json.dumps({'checkpoint_sha256': trainer.sha256(old / 'best.pt')}))
    config.update(output_dir=str(tmp_path / 'specialist/train'),
        mixed_reference_comparison=str(cp), mixed_reference_summary=str(sm),
        mixed_reference_comparison_sha256=trainer.sha256(cp), mixed_reference_summary_sha256=trainer.sha256(sm))
    assert trainer.validate_reference_inputs(config)['checkpoint_sha256'] == trainer.sha256(old / 'best.pt')
    with pytest.raises(ValueError, match='outside the original'):
        trainer.validate_reference_inputs(dict(config, output_dir=str(old)))
    with pytest.raises(ValueError, match='differs from mixed reference: epochs'):
        trainer.validate_reference_inputs(dict(config, epochs=20))
    with pytest.raises(ValueError, match='checksum mismatch'):
        trainer.validate_reference_inputs(dict(config, mixed_reference_summary_sha256='incorrect'))
