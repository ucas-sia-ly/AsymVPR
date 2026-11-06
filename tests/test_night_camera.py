"""Camera ordering, RNG/worker invariance and exact train-state resume."""
from copy import deepcopy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torch.utils.data import DataLoader

from reproduction.camera_augmentation import CameraAugmentation, camera_seed, validate_profile
from reproduction.data import transform
from reproduction.night_camera import NightCamera, camera_identity
from reproduction.night_specialist import NightOnlyQueries
from test_night_only_train import cpu_loop, run


def with_camera(config, tmp_path):
    profile = dict(version=1, status='exploratory', conditions=['night'], probability=.8,
        operations=dict(gain_gamma=dict(gain=[1.1, 1.3], gamma=[.85, .95]),
            bloom=dict(threshold=[.8, .9], strength=[.05, .1], radius=[1., 2.]),
            motion_blur=dict(length=[1, 3], angle=[-20., 20.]),
            noise=dict(shot_peak=[200., 1000.], read_std=[0., .006]),
            jpeg=dict(quality=[75, 100]), saturation=dict(factor=[.6, 1.]),
            vignette=dict(strength=[0., .2])))
    p = tmp_path / 'profile.json'; p.write_text(json.dumps(profile))
    code = Path('reproduction/camera_augmentation.py')
    return dict(config, camera_profile=str(p), camera_profile_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
        camera_implementation_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        camera_order='before_to_tensor_resize_normalize', camera_teacher_target='clean_night')


def test_camera_runs_on_original_pixels_before_tensor_resize_and_preserves_labels(tmp_path, monkeypatch):
    image = Image.fromarray(np.random.default_rng(2).integers(0,256,(43,67,3),dtype='uint8'))
    p = tmp_path / 'night.png'; image.save(p)
    row = dict(night_path=str(p), label=49123, source_index=32101, split='train')
    c = with_camera(dict(seed=2026,image_size=8), tmp_path)
    before = p.read_bytes()
    calls = []
    def camera(self, rgb, actual_row):
        assert actual_row == row and isinstance(rgb, Image.Image)
        calls.append(('camera',rgb.size))
        return Image.new('RGB',rgb.size,(99,66,33)), True
    monkeypatch.setattr(NightCamera,'apply',camera)
    dataset = NightOnlyQueries([row],[row],c,0)
    actual_transform=dataset.transform
    def after(rgb):
        calls.append(('transform',rgb.size))
        assert rgb.getpixel((0,0)) == (99,66,33)
        return actual_transform(rgb)
    dataset.transform=after
    tensor, ni, source, label, applied = dataset[0]
    assert calls == [('camera',(67,43)),('transform',(67,43))]
    assert tensor.shape == (3,8,8) and (ni,source,label,applied)==(0,32101,49123,True)
    torch.testing.assert_close(tensor, transform(8)(Image.new('RGB',(67,43),(99,66,33))),rtol=0,atol=0)
    assert p.read_bytes() == before


def test_rng_and_epoch_replay_are_independent_of_model_and_worker_rng(tmp_path):
    c = with_camera(dict(seed=2026,image_size=16),tmp_path)
    # Force all profile applications so epoch variation always gets exercised.
    profile=json.loads(Path(c['camera_profile']).read_text());profile['probability']=1.
    Path(c['camera_profile']).write_text(json.dumps(profile))
    c['camera_profile_sha256']=hashlib.sha256(Path(c['camera_profile']).read_bytes()).hexdigest()
    rows=[]
    for i in range(6):
        p=tmp_path/f'night{i}.png'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(31,47,3),dtype='uint8')).save(p)
        rows.append(dict(night_path=str(p),source_index=100+i,label=50+i,split='train'))
    dataset=NightOnlyQueries(rows,rows,c,0)
    py, npstate, ts = random.getstate(),np.random.get_state(),torch.get_rng_state()
    first=dataset[0][0];again=dataset[0][0]
    torch.testing.assert_close(first,again,rtol=0,atol=0)
    assert random.getstate()==py
    np.testing.assert_equal(np.random.get_state(),npstate)
    assert torch.equal(torch.get_rng_state(),ts)
    tensors=[]
    for workers in (0,2):
        loader=DataLoader(dataset,batch_size=2,num_workers=workers,
            generator=torch.Generator().manual_seed(6))
        tensors.append(torch.cat([batch[0] for batch in loader]))
    torch.testing.assert_close(tensors[0],tensors[1],rtol=0,atol=0)
    next_epoch=NightOnlyQueries(rows,rows,c,1)
    ni={next_epoch[i][1]:next_epoch[i][0] for i in range(len(rows))}
    assert any(not torch.equal(dataset[i][0],ni[dataset[i][1]]) for i in range(len(rows)))


def test_camera_loop_resumes_exactly_and_logs_real_applied_fraction(cpu_loop,tmp_path):
    config,records,calls=cpu_loop
    config=with_camera(config,tmp_path)
    full,_=run(config,records,tmp_path/'full_camera')
    first,_=run(config,records,tmp_path/'resume_camera',max_steps=2)
    resumed,logger=run(config,records,tmp_path/'resume_camera')
    assert first['step']==2 and resumed['step']==6
    for key,value in full['model'].items():
        torch.testing.assert_close(value,resumed['model'][key],rtol=0,atol=0)
    values=[r['train/camera_applied_fraction'] for r in logger.rows if 'train/loss' in r]
    assert values and all(0<=v<=1 for v in values) and any(v>0 for v in values)
    assert all('train/replay_asym' not in r for r in logger.rows)


def test_profile_and_implementation_hashes_and_order_are_enforced(tmp_path):
    c=with_camera(dict(seed=2026),tmp_path)
    assert camera_identity(c)['order'][1:4]==['camera_degradation','ToTensor','Resize_bicubic_antialias']
    for key,value,match in [('camera_profile_sha256','wrong','profile SHA256'),
            ('camera_implementation_sha256','wrong','implementation SHA256'),
            ('camera_order','after_resize','pre-Tensor'),('camera_teacher_target','degraded','clean-night')]:
        with pytest.raises(ValueError,match=match): camera_identity(dict(c,**{key:value}))


def test_operations_preserve_geometry_and_finite_pixels_and_no_new_light():
    profile=dict(version=1,status='exploratory',conditions=['night'],probability=1.,
        operations={'bloom':dict(threshold=[.8,.8],strength=[.8,.8],radius=[3.,3.])})
    aug=CameraAugmentation(profile)
    black=Image.new('RGB',(31,47))
    np.testing.assert_array_equal(aug.apply(black,seed=7,condition='night'),black)
    profile['operations']={'motion_blur':dict(length=[7,7],angle=[0.,0.])}
    impulse=np.zeros((31,31,3),np.uint8);impulse[15,15]=255
    actual=np.asarray(CameraAugmentation(profile).apply(Image.fromarray(impulse),seed=8,condition='night'))
    assert actual.shape==impulse.shape and np.isfinite(actual).all()
    assert np.count_nonzero(actual[15])>3 and np.count_nonzero(actual[:,15])==3
    profile['operations']={'noise':dict(shot_peak=[0,100],read_std=[0,.01])}
    with pytest.raises(ValueError): validate_profile(profile)
