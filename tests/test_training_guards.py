"""Training reliability checks use small CPU fixtures, never a model run."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from reproduction import run


@pytest.fixture
def cached_experiment(tmp_path):
    cache = tmp_path / 'cache'
    msls = tmp_path / 'msls'
    cache.mkdir()
    msls.mkdir()
    teacher = tmp_path / 'teacher.pt'
    teacher.write_bytes(b'fixed teacher fixture')
    c = dict(cache_dir=str(cache), output_dir=str(tmp_path / 'train'), msls_root=str(msls),
             teacher_weights=str(teacher), image_size=322, descriptor_dim=3,
             precision='float32', epochs=2, batch_size=2)
    meta = dict(signature='train-manifest', num_images=4, num_places=2)
    val_data = (np.array(['ref0', 'ref1', 'query0']), 2, [np.array([0])])
    for name, value in zip(run.MSLS_FILES, [np.array(['ref0', 'ref1']),
            np.array(['query0']), np.array([0]), np.array([[0]])]):
        np.save(msls / name, value)
    np.save(cache / 'paths.npy', np.array(['a', 'b', 'c', 'd'], dtype='S256'))
    np.save(cache / 'labels.npy', np.array([0, 0, 1, 1], dtype='int32'))
    np.save(cache / 'offsets.npy', np.array([0, 2, 4], dtype='int64'))
    for name, rows in [('train_features', 4), ('msls_features', 3), ('centroids', 2), ('variances', 2)]:
        np.save(cache / f'{name}.npy', np.zeros((rows, 3), dtype='float32'))
    for name, rows in [('train_features', 4), ('msls_features', 3)]:
        run.atomic_json(cache / f'{name}.progress.json', dict(rows=rows, total=rows))
    identity = run.provenance(c, meta)
    run.atomic_json(cache / 'provenance.json', identity)
    run.atomic_json(cache / 'bank_complete.json', identity)
    return c, meta, val_data, identity


def test_same_length_msls_reordering_invalidates_cache(cached_experiment):
    c, meta, val_data, identity = cached_experiment
    path = Path(c['msls_root']) / 'msls_val_dbImages.npy'
    np.save(path, np.array(['ref1', 'ref0']))
    assert run.provenance(c, meta) != identity
    with pytest.raises(RuntimeError, match='provenance mismatch'):
        run.validate_cache(c, meta, val_data)


@pytest.mark.parametrize('damage, message', [
    ('incomplete', 'not complete'), ('missing', 'without its feature file'),
    ('shape', 'shape/dtype mismatch'), ('progress', 'Invalid extraction progress'),
    ('bank', 'bank provenance mismatch'), ('variance', 'shape/dtype mismatch'),
])
def test_cache_preflight_rejects_inconsistent_artifacts(cached_experiment, damage, message):
    c, meta, val_data, _ = cached_experiment
    run.validate_cache(c, meta, val_data)
    cache = Path(c['cache_dir'])
    if damage == 'incomplete':
        run.atomic_json(cache / 'train_features.progress.json', dict(rows=2, total=4))
    elif damage == 'missing':
        (cache / 'train_features.npy').unlink()
    elif damage == 'shape':
        np.save(cache / 'train_features.npy', np.zeros((3, 3), dtype='float32'))
    elif damage == 'progress':
        run.atomic_json(cache / 'train_features.progress.json', dict(rows=5, total=4))
    elif damage == 'bank':
        run.atomic_json(cache / 'bank_complete.json', {'different': 'source'})
    else:
        np.save(cache / 'variances.npy', np.zeros((2, 3), dtype='float64'))
    with pytest.raises(RuntimeError, match=message):
        run.validate_cache(c, meta, val_data)


def test_prepare_refuses_unattributed_existing_features(cached_experiment):
    c, meta, _, _ = cached_experiment
    (Path(c['cache_dir']) / 'provenance.json').unlink()
    with pytest.raises(RuntimeError, match='no provenance'):
        run.prepare(c, None, meta)
    assert not (Path(c['cache_dir']) / 'provenance.json').exists()


def test_main_does_not_overwrite_foreign_configuration(tmp_path, monkeypatch):
    out = tmp_path / 'train'
    out.mkdir()
    previous = {'output_dir': str(out), 'epochs': 15}
    original = json.dumps(previous)
    (out / 'config.json').write_text(original)
    config = tmp_path / 'requested.json'
    config.write_text(json.dumps(dict(previous, epochs=16)))
    monkeypatch.setattr('sys.argv', ['run', '--config', str(config)])
    with pytest.raises(RuntimeError, match='different configuration'):
        run.main()
    assert (out / 'config.json').read_text() == original


def test_completed_training_skips_before_cuda_or_dataset_access(cached_experiment, monkeypatch):
    c, _, _, identity = cached_experiment
    out = Path(c['output_dir'])
    out.mkdir()
    run.atomic_json(out / 'config.json', c)
    run.atomic_json(out / 'completed.json', dict(config=c, epochs=2, steps=4,
                    best_msls_r1=90., provenance=identity))
    for name in ['last.pt', 'best.pt', 'epoch_02.pt']:
        (out / name).write_bytes(b'completed artifact fixture')
    config = out.parent / 'requested.json'
    config.write_text(json.dumps(c))
    monkeypatch.setattr('sys.argv', ['run', '--config', str(config)])
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed training must not access data or CUDA')
    monkeypatch.setattr(run, 'manifest', forbidden)
    monkeypatch.setattr(run, 'configure_cuda', forbidden)
    monkeypatch.setattr(run, 'train', forbidden)
    run.main()


def test_failed_epoch_best_save_does_not_advance_last(tmp_path, monkeypatch):
    last = tmp_path / 'last.pt'
    last.write_bytes(b'previous complete resume state')
    original = run.atomic_torch_save
    def fail_best(path, value):
        if Path(path).name == 'best.pt':
            raise OSError('simulated interrupted write')
        original(path, value)
    monkeypatch.setattr(run, 'atomic_torch_save', fail_best)
    model = torch.nn.Linear(1, 1)
    with pytest.raises(OSError, match='interrupted'):
        run.commit_epoch(tmp_path, model, None, None, 1, 2, -1.,
                         {'msls-val/R@1': 80.}, {}, {})
    assert last.read_bytes() == b'previous complete resume state'
    assert not (tmp_path / 'best.pt').exists()


def test_failed_atomic_save_preserves_previous_weights(tmp_path, monkeypatch):
    path = tmp_path / 'best.pt'
    path.write_bytes(b'previous weights')
    def fail_save(value, file):
        file.write(b'partial replacement')
        raise OSError('simulated disk failure')
    monkeypatch.setattr(run.torch, 'save', fail_save)
    with pytest.raises(OSError, match='disk failure'):
        run.atomic_torch_save(path, {})
    assert path.read_bytes() == b'previous weights'
    assert list(tmp_path.glob('*.tmp')) == []


def test_resume_rejects_step_inconsistent_with_batch(cached_experiment):
    c, meta, _, identity = cached_experiment
    state = dict(config=c, provenance=identity, epoch=1, next_batch=1,
                 step=3, scheduler={'last_epoch': 1})
    run.validate_resume_state(state, c, identity, meta)
    state['step'] = 4
    with pytest.raises(RuntimeError, match='epoch/batch/step/scheduler'):
        run.validate_resume_state(state, c, identity, meta)


@pytest.mark.parametrize('capacity_gib', [24, 48])
def test_cuda_allocator_fraction_adapts_to_actual_device(monkeypatch, capacity_gib):
    total = capacity_gib * 1024**3
    monkeypatch.setattr(run.torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(run.torch.cuda, 'get_device_properties', lambda device: SimpleNamespace(total_memory=total))
    monkeypatch.setattr(run.torch.cuda, 'mem_get_info', lambda device: (total, total))
    fractions = []
    monkeypatch.setattr(run.torch.cuda, 'set_per_process_memory_fraction', lambda fraction, **kw: fractions.append(fraction))
    run.configure_cuda()
    assert 0 < fractions[0] < 1
    assert fractions[0] * total <= total - min(4 * 1024**3, total * .1)
