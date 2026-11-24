"""Cache reuse must preserve immutable source bytes and exact model/input identity."""
import json
import os
import numpy as np
import pytest
from reproduction.benchmarks import json_sha256,sha256
from reproduction.night_suite import prepare_cache,reuse_cache,verify_cache


def cache(tmp_path):
    source=tmp_path/'source.npy'
    np.save(source,np.arange(12,dtype=np.float32).reshape(3,4))
    identity=dict(paths={'count':3},descriptor_dim=4,model_sha256='frozen-model')
    source.with_suffix('.provenance.json').write_text(json.dumps(identity))
    progress=dict(rows=3,total=3,identity_sha256=json_sha256(identity),feature_sha256=sha256(source))
    source.with_suffix('.progress.json').write_text(json.dumps(progress))
    return source,identity


def test_promoted_cache_is_exact_read_only_reuse_and_idempotent(tmp_path):
    source,identity=cache(tmp_path)
    checksum=sha256(source)
    target=tmp_path/'active/gallery.npy'
    reuse_cache(source,target,identity)
    assert os.stat(source).st_ino==os.stat(target).st_ino
    assert verify_cache(target,identity)==checksum
    reuse_cache(source,target,identity)
    assert sha256(source)==checksum


def test_identity_or_feature_corruption_is_rejected(tmp_path):
    source,identity=cache(tmp_path)
    with pytest.raises(ValueError,match='identity mismatch'):
        reuse_cache(source,tmp_path/'wrong.npy',dict(identity,model_sha256='another-model'))
    assert not (tmp_path/'wrong.npy').exists()
    with source.open('r+b') as f:
        f.seek(-1,2);b=f.read(1);f.seek(-1,2);f.write(bytes([b[0]^1]))
    with pytest.raises(ValueError,match='checksum mismatch'):
        verify_cache(source,identity)


def test_optional_reuse_falls_back_to_extraction_and_reuses_completed_target(tmp_path):
    source, identity = cache(tmp_path)
    target = tmp_path / 'new' / 'gallery.npy'
    assert prepare_cache(target, identity) is None
    assert prepare_cache(target, identity, tmp_path / 'removed-archive.npy') is None
    note = prepare_cache(target, identity, source)
    assert note['feature_sha256'] == sha256(source)
    source.unlink()
    assert prepare_cache(target, identity)['feature_sha256'] == sha256(target)


def test_partial_target_resumes_without_replacing_it_from_source(tmp_path):
    source, identity = cache(tmp_path)
    target = tmp_path / 'target.npy'
    reuse_cache(source, target, identity)
    progress = json.loads(target.with_suffix('.progress.json').read_text())
    progress['rows'] = 1
    target.with_suffix('.progress.json').write_text(json.dumps(progress))
    assert prepare_cache(target, identity, source) is None
    assert json.loads(target.with_suffix('.progress.json').read_text())['rows'] == 1


def test_orphan_sidecar_is_not_overwritten_by_optional_reuse(tmp_path):
    source, identity = cache(tmp_path)
    target = tmp_path / 'orphan.npy'
    sidecar = target.with_suffix('.provenance.json')
    sidecar.write_text('uncommitted previous evidence')
    with pytest.raises(ValueError, match='Incomplete cache metadata'):
        prepare_cache(target, identity, source)
    assert sidecar.read_text() == 'uncommitted previous evidence'
    assert not target.exists()


def test_manifest_only_needs_no_retired_experiment(tmp_path, monkeypatch):
    from reproduction import night_suite
    from types import SimpleNamespace
    monkeypatch.chdir(tmp_path)
    desc = {'protocol': 'fixture', 'manifest_sha256': 'fixture',
            'references': {'count': 10}, 'queries': {'count': 2}}
    monkeypatch.setattr(night_suite, 'BUILDERS', {
        'fixture': lambda: SimpleNamespace(name='fixture', describe=lambda: desc)})
    monkeypatch.setattr('sys.argv', ['night_suite', '--datasets', 'fixture', '--manifest-only'])
    night_suite.main()
    out = tmp_path / 'results/mixed_night_suite_v1'
    assert json.loads((out / 'manifests.json').read_text()) == {'fixture': desc}
    assert not (out / 'summary.json').exists()
