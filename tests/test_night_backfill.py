"""CPU checks for protocol integrity, safe result reuse and evaluation-only mode."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from reproduction.benchmarks import Benchmark, json_sha256, sha256
from reproduction.evaluate import feature_identity
from reproduction.extra_night_benchmarks import load_protocol
from reproduction import night_backfill as backfill


def seal(path, body):
    path.write_text(json.dumps(dict(body, record_sha256=json_sha256(body))))


def test_protocol_checks_receipt_sources_and_original_image_bytes(tmp_path):
    source = tmp_path / 'poses.txt'; source.write_text('verified source')
    refs = [tmp_path / 'ref.jpg']; query = tmp_path / 'night.jpg'
    for p in refs + [query]:
        p.write_bytes(b'original pixels')
    record = dict(name='night', root=str(tmp_path), references=list(map(str, refs)),
        queries=[str(query)], positives=[[0]], protocol={'condition': 'night'},
        source_files={str(source): sha256(source)}, image_sha256={str(query): sha256(query)})
    path = tmp_path / 'protocol.json'; seal(path, record)
    got = load_protocol(path)
    assert got.name == 'night' and got.positives[0].tolist() == [0]
    query.write_bytes(b'changed image')
    with pytest.raises(ValueError, match='image mapping bytes changed'):
        load_protocol(path)
    query.write_bytes(b'original pixels'); source.write_text('changed source')
    with pytest.raises(ValueError, match='source changed'):
        load_protocol(path)
    source.write_text('verified source')
    changed = dict(record, positives=[[1]]); seal(path, changed)
    with pytest.raises(ValueError, match='Invalid or empty ground truth'):
        load_protocol(path)
    path.write_text(path.read_text().replace('"night"', '"day"'))
    with pytest.raises(ValueError, match='receipt checksum'):
        load_protocol(path)


def tiny_benchmark(root):
    refs = [root / f'r{i}.jpg' for i in range(10)]
    qs = [root / 'q.jpg']
    for p in refs + qs:
        p.write_bytes(b'fixture image')
    return Benchmark('tiny_night', root, refs, qs, [np.array([0])], {'condition': 'night'}, {})


def test_actual_predictions_define_recall_and_invalid_indices_are_rejected(tmp_path):
    b = tiny_benchmark(tmp_path)
    pred = np.array([[9, 0, 1, 2, 3, 4, 5, 6, 7, 8]])
    metrics, hits = backfill.checked_metrics(pred, b)
    assert metrics == {'R@1': 0., 'R@5': 100., 'R@10': 100.}
    assert hits.tolist() == [[False, True, True]]
    pred[0, 0] = 10
    with pytest.raises(ValueError, match='Top-10'):
        backfill.checked_metrics(pred, b)


def write_cache(path, identity):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((identity['paths']['count'], identity['descriptor_dim']), np.float32))
    path.with_suffix('.provenance.json').write_text(json.dumps(identity))
    path.with_suffix('.progress.json').write_text(json.dumps(dict(rows=identity['paths']['count'],
        total=identity['paths']['count'], identity_sha256=json_sha256(identity), feature_sha256=sha256(path))))


@pytest.fixture
def prior(tmp_path):
    b = tiny_benchmark(tmp_path)
    src = tmp_path / 'source'; dest = src / b.name; dest.mkdir(parents=True)
    teacher, checkpoint = 'a'*64, 'b'*64
    config = json.loads(Path('configs/boq_b2.json').read_text())
    row = dict(checkpoint_sha256=checkpoint, teacher_sha256=teacher, query_count=1,
               gallery_count=10, metrics_percent={'R@1': 0., 'R@5': 100., 'R@10': 100.})
    for role, paths, model, kind in [('gallery', b.references, teacher, 'gallery_teacher'),
                                   ('query', b.queries, checkpoint, 'query_student')]:
        path = src / 'cache' / f'{role}.npy'
        identity = feature_identity(paths, config, model, kind)
        write_cache(path, identity)
        row[f'{role}_cache'] = dict(path=str(path), feature_sha256=sha256(path))
    predicted = dest / 'original_top10.npy'
    np.save(predicted, np.array([[9, 0, 1, 2, 3, 4, 5, 6, 7, 8]]))
    row['predictions_sha256'] = sha256(predicted)
    summary = dict(status='complete', precision='float32', tf32=False, teacher_sha256=teacher,
        protocols={b.name: b.describe()}, models={'original': {'sha256': checkpoint}},
        results={b.name: {'original': row}})
    (src / 'summary.json').write_text(json.dumps(summary))
    return b, src, row, summary


def test_prior_result_is_recounted_and_source_artifacts_are_unchanged(prior, tmp_path):
    b, source, row, summary = prior
    before = sha256(source / 'summary.json')
    got = backfill.reuse_result(b, 'original', {'checkpoint_sha256': row['checkpoint_sha256']},
        row['teacher_sha256'], source, tmp_path / 'new')
    assert got['metrics_percent'] == row['metrics_percent']
    assert got['evaluation_reuse']['recomputed_hits'] is True
    assert sha256(source / 'summary.json') == before
    assert np.load(tmp_path / 'new/original_hits.npy').tolist() == [[False, True, True]]


@pytest.mark.parametrize('damage', ['teacher', 'protocol', 'predictions', 'features'])
def test_wrong_prior_result_is_never_promoted(prior, tmp_path, damage):
    b, source, row, summary = prior
    if damage == 'teacher':
        summary['teacher_sha256'] = 'other'
    elif damage == 'protocol':
        summary['protocols'][b.name]['ground_truth_sha256'] = 'other'
    elif damage == 'predictions':
        np.save(source / b.name / 'original_top10.npy', np.zeros((1, 10), np.int64))
    else:
        p = Path(row['query_cache']['path'])
        with p.open('r+b') as f:
            f.seek(-1, 2); f.write(b'\xff')
    (source / 'summary.json').write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        backfill.reuse_result(b, 'original', {'checkpoint_sha256': row['checkpoint_sha256']},
            row['teacher_sha256'], source, tmp_path / 'new')
    assert not (tmp_path / 'new/original_top10.npy').exists()


def test_manifest_only_has_no_cuda_work_and_restricts_scope_to_retained(tmp_path, monkeypatch):
    b = tiny_benchmark(tmp_path)
    registry = dict(models={k: {} for k in ('original', 'mixed', 'mixed_camera')},
                    output=str(tmp_path / 'out'), cache=str(tmp_path / 'cache'))
    path = tmp_path / 'registry.json'; path.write_text(json.dumps(registry))
    monkeypatch.setattr(sys, 'argv', ['night_backfill', '--registry', str(path), '--manifest-only'])
    monkeypatch.setattr(backfill, 'BUILDERS', {'tiny': lambda: b})
    monkeypatch.setattr(backfill, 'EXTRA_BUILDERS', {})
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: pytest.fail('manifest-only touched CUDA'))
    backfill.main()
    assert (tmp_path / 'out/manifests.json').exists()
    registry['models']['retired'] = {}
    path.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match='restricted to the three retained'):
        backfill.main()
