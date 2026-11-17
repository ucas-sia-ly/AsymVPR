import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from reproduction import night_subset
from reproduction.benchmarks import json_sha256
from reproduction.evaluate import atomic_json, sha256, feature_identity, open_feature_cache
from reproduction.night_data import NightDataIntegrityError
from reproduction.night_train import PairedReplay
from test_night_train import replay_inputs


def pairs(root):
    rows=[]
    for i,split in enumerate(['val','train','val']):
        row=dict(sample_id=str(i),label=i+1,split=split,source_index=i,source_dimensions=[8,8])
        for role in ['source','night']:
            path=root/f'{role}{i}.png';Image.new('RGB',(8,8),(30+20*i,70,90)).save(path)
            st=path.stat()
            row.update({role+'_path':str(path),role+'_sha256':sha256(path),
                        role+'_size':st.st_size,role+'_mtime_ns':st.st_mtime_ns})
        rows.append(row)
    return rows


def test_remaining_audit_keeps_original_split_and_parent_feature_indices(tmp_path):
    rows=pairs(tmp_path);Path(rows[0]['night_path']).unlink()
    kept,excluded,mapping=night_subset.audit_remaining(rows,workers=1)
    assert mapping.tolist()==[1,2]
    assert [(r['split'],r['source_index'],r['parent_teacher_index']) for r in kept]==[('train',1,1),('val',2,2)]
    assert excluded[0]['reason']=='missing_night_image' and excluded[0]['split']=='val'


def test_remaining_audit_rejects_changed_image_bytes(tmp_path):
    rows=pairs(tmp_path);Image.new('RGB',(8,8),'red').save(rows[1]['night_path'])
    with pytest.raises(NightDataIntegrityError,match='SHA256 differs'):
        night_subset.audit_remaining(rows,workers=1)


def test_feature_subset_matches_exact_parent_rows_and_retains_deleted_val_exclusion(tmp_path,monkeypatch):
    rows=pairs(tmp_path)
    parent=tmp_path/'parent';parent.mkdir()
    (parent/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    atomic_json(parent/'summary.json',dict(input_config={},total=3,train=1,val=2))
    teacher=tmp_path/'teacher.pt';teacher.write_bytes(b'teacher identity')
    source_cache=tmp_path/'teacher.npy'
    config=dict(input_dir=str(parent),night_teacher_cache=str(source_cache),
                teacher_weights=str(teacher),image_size=8,descriptor_dim=3)
    identity=night_subset.parent_feature_identity(rows,config)
    assert identity==feature_identity([Path(r['night_path']) for r in rows],config,sha256(teacher),'night_teacher')
    np.save(source_cache,np.eye(3,dtype=np.float32))
    atomic_json(source_cache.with_suffix('.provenance.json'),identity)
    atomic_json(source_cache.with_suffix('.progress.json'),dict(rows=3,total=3,
        identity_sha256=json_sha256(identity),feature_sha256=sha256(source_cache)))
    parent_config=tmp_path/'config.json';atomic_json(parent_config,config)
    Path(rows[0]['night_path']).unlink()
    monkeypatch.setattr(night_subset,'_validate_frozen',lambda *args:None)
    root=tmp_path/'remaining'
    result=night_subset.build_remaining(root,parent_config)
    assert (result['total'],result['train'],result['val'])==(2,1,1)
    assert result['derivation']['replay_excluded_place_ids']==[1,3]
    actual=np.load(root/'cache/night_teacher.npy')
    np.testing.assert_array_equal(actual,np.eye(3,dtype=np.float32)[[1,2]])
    kept=[json.loads(s) for s in (root/'inputs/records.jsonl').open()]
    identity=feature_identity([Path(r['night_path']) for r in kept],config,sha256(teacher),'night_teacher')
    features,done=open_feature_cache(root/'cache/night_teacher.npy',identity)
    assert done==2
    del features
    assert len((parent/'records.jsonl').read_text().splitlines())==3


def test_original_validation_places_stay_excluded_even_if_all_night_images_removed(replay_inputs):
    rows,all_rows,config,labels=replay_inputs
    # Label 2 is normally eligible for replay in this fixture; explicitly preserve it as old holdout.
    dataset=PairedReplay(rows,all_rows,dict(config,replay_excluded_place_ids=[2]),0)
    assert not np.isin(labels[dataset.replay],[2,7]).any()
