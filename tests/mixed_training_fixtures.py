"""Small CPU-only fixture for retained mixed training resume tests."""
from copy import deepcopy
import numpy as np
from PIL import Image
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from reproduction import night_train as trainer

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
    monkeypatch.setattr(trainer, 'STOP', False)
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
        night_batch_size=3, replay_batch_size=3, epochs=2, seed=2026, image_size=8,
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

    monkeypatch.setattr(trainer, 'validation', validation)
    return config, all_records, calls

class Logger:
    def __init__(self):
        self.rows = []

    def log(self, step, **values):
        self.rows.append(dict(step=step, **values))
