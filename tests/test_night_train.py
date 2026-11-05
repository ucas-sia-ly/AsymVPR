"""CPU-only scheduling, selection and complete-state resume checks."""
from copy import deepcopy
import random

import numpy as np
from PIL import Image
import pytest
import torch

from reproduction import night_train


@pytest.fixture
def replay_inputs(tmp_path):
    cache = tmp_path / 'cache'
    images = tmp_path / 'gsv'
    cache.mkdir()
    images.mkdir()
    labels = np.array([2, 2, 7, 7, 11, 11], dtype='int32')
    names = [f'day_{index}.png' for index in range(len(labels))]
    np.save(cache / 'paths.npy', np.array(names, dtype='S256'))
    np.save(cache / 'labels.npy', labels)
    for index, name in enumerate(names):
        Image.new('RGB', (8, 8), (20 + index, 60, 90)).save(images / name)
    records = []
    for index in range(8):
        path = tmp_path / f'night_{index}.png'
        Image.new('RGB', (8, 8), (80, 10 + index, 20)).save(path)
        source = [0, 1, 4, 5][index % 4]
        records.append(dict(night_path=str(path), source_index=source,
                            label=int(labels[source]), split='train'))
    val = dict(night_path=str(tmp_path / 'heldout.png'), source_index=2, label=7, split='val')
    # The all-record order deliberately differs from the train-record order.
    all_records = [val, *reversed(records)]
    config = dict(base_cache=str(cache), gsv_root=str(images), seed=2026, image_size=8)
    return records, all_records, config, labels


def test_paired_replay_schedule_is_deterministic_and_excludes_heldout_places(replay_inputs):
    records, all_records, config, labels = replay_inputs
    first = night_train.PairedReplay(records, all_records, config, epoch=0)
    repeated = night_train.PairedReplay(records, all_records, config, epoch=0)
    next_epoch = night_train.PairedReplay(records, all_records, config, epoch=1)
    np.testing.assert_array_equal(first.order, repeated.order)
    np.testing.assert_array_equal(first.replay, repeated.replay)
    np.testing.assert_array_equal(np.sort(first.order), np.arange(len(records)))
    assert len(first.replay) == len(records)
    assert not np.isin(labels[first.replay], [7]).any()
    assert not (np.array_equal(first.order, next_epoch.order)
                and np.array_equal(first.replay, next_epoch.replay))


def test_replay_item_keeps_global_labels_and_all_snapshot_teacher_row(replay_inputs):
    records, all_records, config, labels = replay_inputs
    dataset = night_train.PairedReplay(records, all_records, config, epoch=0)
    for index in range(len(dataset)):
        night, day, teacher_row, source, label, replay_row = dataset[index]
        expected = records[int(dataset.order[index])]
        assert night.shape == day.shape == (3, 8, 8)
        assert all_records[teacher_row]['night_path'] == expected['night_path']
        assert (source, label) == (expected['source_index'], expected['label'])
        assert label == labels[source] and label in {2, 11}
        assert labels[replay_row] in {2, 11}


def test_replay_fallback_val_manifest_also_excludes_heldout_places(replay_inputs, monkeypatch):
    records, all_records, config, labels = replay_inputs
    val_rows = [row for row in all_records if row['split'] == 'val']
    untagged = [{key: value for key, value in row.items() if key != 'split'} for row in all_records]
    calls = []
    def val_loader(actual_config, split):
        assert actual_config == config
        calls.append(split)
        return val_rows
    monkeypatch.setattr(night_train, 'load_records', val_loader)
    dataset = night_train.PairedReplay(records, untagged, config, epoch=0)
    assert calls == ['val']
    assert not np.isin(labels[dataset.replay], [7]).any()


@pytest.mark.parametrize('current_day, eligible', [(92., True), (91., True), (90.9999, False)])
def test_checkpoint_selection_applies_day_guard_at_inclusive_floor(current_day, eligible):
    baseline = {'msls': {'metrics_percent': {'R@1': 92.}}}
    metrics = {'msls': {'metrics_percent': {'R@1': current_day}},
               'svox_dev': {'metrics_percent': {'R@1': 48.5}},
               'synthetic': {'metrics_percent': {'R@1': 99.}}}
    actual_eligible, score, delta = night_train.eligible_selection(metrics, baseline, 1.)
    assert actual_eligible is eligible
    assert score == 48.5  # Real-night development, not synthetic R@1, selects weights.
    assert delta == pytest.approx(current_day - 92.)


@pytest.fixture
def cpu_checkpoint(monkeypatch):
    random.seed(51)
    np.random.seed(51)
    torch.manual_seed(51)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    restored_cuda = []
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda values: restored_cuda.append(values))
    config = dict(epochs=2, lr=.01)
    lineage = dict(init_sha256='fixed-init', manifest_sha256='fixed-snapshot')
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .9 ** epoch)
    optimizer.zero_grad()
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    scheduler.step()
    best = dict(score=45., epoch=0, step=0, adapted=False)
    baseline = {'msls': {'metrics_percent': {'R@1': 92.}}}
    state = deepcopy(night_train.checkpoint_payload(model, optimizer, scheduler,
        epoch=1, next_batch=2, step=7, best=best, baseline=baseline,
        config=config, lineage=lineage))
    return state, config, lineage, restored_cuda


def restore_cpu(state, config, lineage):
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: .9 ** epoch)
    restored = night_train.restore_state(state, model, optimizer, scheduler, config, lineage,
                                         steps_per_epoch=5)
    return restored, model, optimizer, scheduler


def test_complete_cpu_checkpoint_restores_parameters_optimizer_scheduler_and_rng(cpu_checkpoint):
    state, config, lineage, cuda_calls = cpu_checkpoint
    expected_random = (random.random(), float(np.random.rand()), torch.rand(2))
    restored, model, optimizer, scheduler = restore_cpu(state, config, lineage)
    assert restored == (1, 2, 7, state['best'], state['baseline'])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state['model'][name], rtol=0, atol=0)
    assert optimizer.param_groups[0]['lr'] == state['optimizer']['param_groups'][0]['lr']
    assert len(optimizer.state) == len(state['optimizer']['state'])
    assert scheduler.state_dict() == state['scheduler']
    assert random.random() == expected_random[0]
    assert float(np.random.rand()) == expected_random[1]
    torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)
    assert cuda_calls == [[]]


@pytest.mark.parametrize('damage', ['step', 'scheduler', 'finished_with_batch', 'float_counter', 'bool_counter'])
def test_resume_rejects_inconsistent_or_non_integer_counters(cpu_checkpoint, damage):
    state, config, lineage, cuda_calls = cpu_checkpoint
    if damage == 'step':
        state['step'] += 1
    elif damage == 'scheduler':
        state['scheduler']['last_epoch'] += 1
    elif damage == 'finished_with_batch':
        state.update(epoch=config['epochs'], next_batch=1, step=config['epochs'] * 5 + 1)
        state['scheduler']['last_epoch'] = config['epochs']
    elif damage == 'float_counter':
        state.update(epoch=1., next_batch=2., step=7.)
    else:
        state.update(epoch=True, next_batch=2, step=7)
    with pytest.raises(RuntimeError, match='resume counters/scheduler'):
        restore_cpu(state, config, lineage)
    assert cuda_calls == []


@pytest.mark.parametrize('field', ['config', 'lineage'])
def test_resume_refuses_different_experiment_or_input_lineage(cpu_checkpoint, field):
    state, config, lineage, cuda_calls = cpu_checkpoint
    state[field] = {**state[field], 'changed': True}
    with pytest.raises(RuntimeError, match='config or immutable input lineage'):
        restore_cpu(state, config, lineage)
    assert cuda_calls == []
