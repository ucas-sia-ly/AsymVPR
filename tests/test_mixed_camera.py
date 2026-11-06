"""Mixed replay stays clean and reproducible when only nights are degraded."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from reproduction import night_train
from reproduction.benchmarks import json_sha256
from reproduction.evaluate import sha256
from reproduction.mixed_camera import report_completed, validate_config
from reproduction.night_finalize import _sealed
from test_night_camera import with_camera
from test_night_only_train import cpu_loop, Logger, TinyStudent
from test_night_train import replay_inputs


def test_only_night_changes_and_replay_schedule_labels_and_day_pixels_stay_exact(replay_inputs,tmp_path):
    records,all_rows,config,labels=replay_inputs
    augmented=with_camera(config,tmp_path)
    clean=night_train.PairedReplay(records,all_rows,config,0)
    camera=night_train.PairedReplay(records,all_rows,augmented,0)
    np.testing.assert_array_equal(clean.order,camera.order)
    np.testing.assert_array_equal(clean.replay,camera.replay)
    changed=0
    for i in range(len(camera)):
        a,b=clean[i],camera[i]
        assert len(a)==6 and len(b)==7 and a[2:]==b[2:6]
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)
        torch.testing.assert_close(b[0],camera[i][0],rtol=0,atol=0)
        changed+=not torch.equal(a[0],b[0])
        assert not np.isin(labels[b[5]],[7])
    assert changed>0


class MixedStudent(TinyStudent):
    def forward(self,images):
        assert len(images)<=6
        return F.normalize(self.aggregator(self.backbone(images)),dim=-1)


def test_actual_mixed_camera_loop_resumes_exactly_keeps_balanced_loss_and_day_guard(cpu_loop,tmp_path,monkeypatch):
    from reproduction import night_only_train
    config,all_rows,calls=cpu_loop
    records=list(reversed(all_rows))
    config=with_camera(dict(config,replay_batch_size=3,batch_size=6,
        gate_cos_min=.2,gate_cos_span=.5,max_night_weight=.5,max_msls_drop_pp=1.),tmp_path)
    cache=Path(config['base_cache']);days=tmp_path/'days';days.mkdir()
    names=[f'day{i}.png' for i in range(8)]
    for name in names:Image.new('RGB',(8,8),(150,120,90)).save(days/name)
    np.save(cache/'paths.npy',np.asarray(names,dtype='S256'))
    np.save(cache/'labels.npy',np.asarray([3 if i%2==0 else 1 for i in range(8)],dtype='int32'))
    config['gsv_root']=str(days)
    monkeypatch.setattr(night_train,'EfficientViTBoQ',MixedStudent)
    monkeypatch.setattr(night_train,'load_records',lambda c,split: [] if split=='val' else records)
    monkeypatch.setattr(night_train,'load_inference_checkpoint',night_only_train.load_inference_checkpoint)

    def execute(folder,max_steps=None):
        folder.mkdir(exist_ok=True)
        c=dict(config,output_dir=str(folder));logger=Logger()
        night_train.seed_all(2026)
        night_train.train(c,all_rows,{'base_provenance':{}},logger,max_steps)
        return torch.load(folder/'last.pt',map_location='cpu',weights_only=False),logger

    whole,logs=execute(tmp_path/'mixed_whole')
    first,_=execute(tmp_path/'mixed_resume',2)
    resumed,_=execute(tmp_path/'mixed_resume')
    assert first['step']==2 and resumed['step']==6
    for key,value in whole['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
    for row in logs.rows:
        if 'train/loss' in row:
            assert row['train/loss']==pytest.approx(.5*(row['train/night_asym']+row['train/replay_asym']),rel=1e-6,abs=1e-6)
            assert 0<=row['train/camera_applied_fraction']<=1
    # The fixture deliberately collapses MSLS by 20pp/epoch: original guard must veto.
    assert resumed['best']['epoch']==0 and not resumed['best']['adapted']
    assert json.loads((tmp_path/'mixed_resume/epoch_02_metrics.json').read_text())['eligible_for_selection'] is False


@pytest.mark.parametrize('change',[{'query_domain':'night_only'},{'camera_student_branch':'day'},
    {'selection_policy':'svox_night_r1'},{'batch_size':32}])
def test_invalid_mixed_camera_contract_fails_before_reference_io(change):
    config=json.loads(Path('configs/boq_b2_mixed_camera_v1.json').read_text())
    with pytest.raises(ValueError,match='Mixed-camera requires'):
        validate_config(dict(config,**change))


@pytest.mark.parametrize('key,value',[('gate_cos_min',.1),('max_night_weight',1.),('max_msls_drop_pp',None)])
def test_original_positive_mix_and_day_guard_cannot_change(key,value):
    config=json.loads(Path('configs/boq_b2_mixed_camera_v1.json').read_text())
    original=json.loads(Path('configs/boq_b2_night_v1.json').read_text())
    with patch('reproduction.night_only_train.validate_reference_inputs',return_value={}), \
         patch('reproduction.mixed_camera._read_sealed',return_value={'training_config':original}):
        with pytest.raises(ValueError,match='controlled field: '+key):
            validate_config(dict(config,**{key:value}))


def test_finalize_extra_report_is_only_used_for_mixed_camera():
    with patch('reproduction.night_finalize.finalize',return_value={'done':True}), \
         patch('reproduction.mixed_camera.report_completed') as report, \
         patch('torch.cuda.is_initialized',return_value=False):
        night_train.finalize_completed({'camera_profile':'fixture','query_domain':'night_only'})
        report.assert_not_called()
        night_train.finalize_completed({'camera_profile':'fixture','query_domain':'mixed'})
        report.assert_called_once()


def test_mixed_camera_report_uses_verified_comparison_and_checks_seal(tmp_path):
    root=tmp_path/'mixed_camera';root.mkdir()
    config={'output_dir':str(root/'train')}
    comparison=dict(svox_dev=dict(mixed_percent={'R@1':89.5},selected_percent={'R@1':90.1}),
        five_benchmark_comparison={'msls':dict(initial_percent={'R@1':91.7},final_percent={'R@1':91.8},
            difference_percentage_points={'R@1':.1})},tokyo_night={'available':False})
    result=dict(status='complete',identity={'fixture':True},
        comparison=dict(selected={'epoch':3},mixed_reference_comparison=comparison),outputs_sha256={})
    result['identity_sha256']=json_sha256(result['identity'])
    (root/'finalization.json').write_text(json.dumps(_sealed(result)))
    with patch('reproduction.mixed_camera.validate_config',return_value={'fixture':'camera'}):
        first=report_completed(config,result)
        second=report_completed(config,result)
        assert first==second
        report=root/'MIXED_CAMERA_REPORT.md'
        text=report.read_text()
        assert '32 night + 32 day' in text and 'ToTensor → Resize' in text
        assert 'MSLS' in text and '0.6000 pp' in text
        report.write_text('corrupt')
        with pytest.raises(ValueError,match='sealed outputs changed'):
            report_completed(config,result)
