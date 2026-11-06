"""CPU-only checks for the separate night-query specialist experiment."""
from copy import deepcopy
import math

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from reproduction.asym_model import asym_loss
from reproduction.night_specialist import (NightOnlyQueries, night_only_objective,
    select_night_best, validate_night_only_config)


def configuration():
    return dict(query_domain='night_only', precision='float32', night_target='teacher_night',
                selection_policy='svox_night_r1', night_batch_size=32,
                batch_size=32, replay_batch_size=0, max_msls_drop_pp=None,
                epochs=10, seed=2026, image_size=8)


@pytest.fixture
def query_rows(tmp_path):
    records = []
    for index in range(8):
        night = tmp_path / f'night_{index}.png'
        Image.new('RGB', (8, 8), (20 + index * 12, 70, 120)).save(night)
        records.append(dict(night_path=str(night), source_index=100 + index,
            source_path=str(tmp_path / 'unavailable_day' / f'{index}.jpg'),
            label=53 if index % 2 else 607, split='train'))
    val = dict(night_path=str(tmp_path / 'heldout.png'), source_index=19, label=900, split='val')
    return records, [val, *reversed(records)]


def test_only_night_files_are_opened_and_global_indices_are_preserved(query_rows, monkeypatch):
    records, all_records = query_rows
    config = dict(configuration(), base_cache='/unavailable/base/cache', gsv_root='/unavailable/day/images')
    dataset = NightOnlyQueries(records, all_records, config, epoch=0)
    original_open, opened = Image.open, []
    def record_open(path, *args, **kwargs):
        opened.append(str(path))
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Image, 'open', record_open)
    for index in range(len(dataset)):
        night, teacher_row, source, label = dataset[index]
        expected = records[int(dataset.order[index])]
        assert night.shape == (3, 8, 8)
        assert all_records[teacher_row] == expected
        assert source == expected['source_index']
        assert label == expected['label'] and label in {53, 607}
    assert opened == [records[int(index)]['night_path'] for index in dataset.order]


def test_night_order_matches_prior_seeded_permutation(query_rows):
    records, all_records = query_rows
    config = configuration()
    first = NightOnlyQueries(records, all_records, config, epoch=0)
    repeat = NightOnlyQueries(records, all_records, config, epoch=0)
    second = NightOnlyQueries(records, all_records, config, epoch=1)
    expected = np.random.default_rng(config['seed']).permutation(len(records))
    np.testing.assert_array_equal(first.order, expected)
    np.testing.assert_array_equal(repeat.order, expected)
    assert not np.array_equal(first.order, second.order)


def test_student_batchnorm_receives_only_night_queries(query_rows):
    records, all_records = query_rows
    dataset = NightOnlyQueries(records, all_records, configuration(), epoch=0)
    night, teacher_row, source, label = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    bn = torch.nn.BatchNorm2d(3, momentum=1.)
    observed = []
    handle = bn.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
    bn.train()
    bn(night)
    handle.remove()
    assert len(observed) == 1 and observed[0].shape == (4, 3, 8, 8)
    torch.testing.assert_close(observed[0], night)
    torch.testing.assert_close(bn.running_mean, night.mean((0, 2, 3)))
    assert int(bn.num_batches_tracked) == 1


def loss_inputs():
    generator = torch.Generator().manual_seed(7)
    q = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    night = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    day = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    kwargs = dict(teacher_night=night, teacher_source_day=day,
        night_place_ids=torch.tensor([0, 3, 6]),
        centroids=(torch.randn(7, 8, generator=generator, dtype=torch.float64) / 10).requires_grad_(),
        variances=(torch.rand(7, 8, generator=generator, dtype=torch.float64) / 100).requires_grad_(),
        temperature=.05, gamma=15., chunk_size=2)
    return q, kwargs


def test_pure_night_eq8_has_no_half_scaling_or_teacher_bank_gradients():
    q, kwargs = loss_inputs()
    actual, metrics = night_only_objective(q, **kwargs)
    expected = asym_loss(q, kwargs['teacher_night'].detach(), kwargs['night_place_ids'],
        kwargs['centroids'].detach(), kwargs['variances'].detach(), chunk_size=2)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm() > 0
    for key in ('teacher_night', 'teacher_source_day', 'centroids', 'variances'):
        assert kwargs[key].grad is None
    assert 'replay_asym' not in metrics
    for value in metrics.values():
        assert value.ndim == 0 and torch.isfinite(value) and not value.requires_grad


def test_changing_or_omitting_day_diagnostics_cannot_change_loss_or_gradient():
    q, kwargs = loss_inputs()
    initial, initial_metrics = night_only_objective(q, **kwargs)
    initial_grad, = torch.autograd.grad(initial, q)
    changed = dict(kwargs, teacher_source_day=-kwargs['teacher_source_day'])
    other, other_metrics = night_only_objective(q, **changed)
    other_grad, = torch.autograd.grad(other, q)
    absent = dict(kwargs, teacher_source_day=None)
    no_day, no_day_metrics = night_only_objective(q, **absent)
    no_day_grad, = torch.autograd.grad(no_day, q)
    torch.testing.assert_close(initial, other, rtol=0, atol=0)
    torch.testing.assert_close(initial, no_day, rtol=0, atol=0)
    torch.testing.assert_close(initial_grad, other_grad, rtol=0, atol=0)
    torch.testing.assert_close(initial_grad, no_day_grad, rtol=0, atol=0)
    torch.testing.assert_close(other_metrics['teacher_daynight_cos'], -initial_metrics['teacher_daynight_cos'])
    assert set(no_day_metrics) == {'night_asym', 'qnight_teachernight_cos'}


def test_selection_uses_real_night_dev_with_no_daytime_floor():
    baseline = {'msls': {'metrics_percent': {'R@1': 92.}}}
    metrics = {'msls': {'metrics_percent': {'R@1': 20.}},
               'svox_dev': {'metrics_percent': {'R@1': 51.}},
               'synthetic': {'metrics_percent': {'R@1': 99.}}}
    eligible, score, delta = select_night_best(metrics, baseline)
    assert eligible is True and score == 51. and delta == -72.


def test_night_budget_matches_prior_night_exposure_and_update_count():
    config = configuration()
    validate_night_only_config(config)
    count = 29050
    rows = [dict(night_path=f'/not-opened/night_{index}.png', source_index=index,
                 label=100 + index // 2, split='train') for index in range(count)]
    dataset = NightOnlyQueries(rows, rows, config, epoch=0)
    assert len(dataset) == count
    assert math.ceil(len(dataset) / config['night_batch_size']) * config['epochs'] == 9080
    assert len(dataset) * config['epochs'] == 290500
    assert config['batch_size'] == config['night_batch_size'] == 32


@pytest.mark.parametrize('key, value', [
    ('query_domain', 'mixed'), ('precision', 'float16'), ('night_target', 'mixed_teacher'),
    ('selection_policy', 'msls_guard'), ('night_batch_size', 0), ('night_batch_size', True),
    ('batch_size', 64), ('replay_batch_size', 32), ('replay_batch_size', False),
    ('max_msls_drop_pp', 1.), ('epochs', 0),
])
def test_configuration_rejects_accidental_day_constraints_or_wrong_budget(key, value):
    config = deepcopy(configuration())
    config[key] = value
    with pytest.raises(ValueError):
        validate_night_only_config(config)


def test_heldout_rows_are_not_accepted_for_training(query_rows):
    _, all_records = query_rows
    with pytest.raises(ValueError, match='Held-out'):
        NightOnlyQueries([all_records[0]], all_records, configuration(), epoch=0)
