"""Cache reuse must preserve immutable source bytes and exact model/input identity."""
import json
import os
import numpy as np
import pytest
from reproduction.benchmarks import json_sha256,sha256
from reproduction.night_suite import reuse_cache,verify_cache


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
