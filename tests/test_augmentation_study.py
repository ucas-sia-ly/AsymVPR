import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from reproduction.augmentation_study import StudyCamera, study_identity, legacy_profile, validate_profile
from reproduction.camera_augmentation import CameraAugmentation, camera_seed
from reproduction import night_train
from test_night_train import replay_inputs


def config(kind, p=1., s=1.):
    return dict(seed=2026, image_size=322, augmentation_study=dict(version=1,kind=kind,probability=p,severity=s))


def image():
    return Image.fromarray(np.random.default_rng(7).integers(0,256,(384,512,3),dtype=np.uint8))


@pytest.mark.parametrize('kind', ['white_balance','local_exposure','resolution','defocus','legacy'])
def test_deterministic_and_no_global_rng_consumption(kind):
    im=image(); row=dict(night_path='sample', source_index=11)
    random.seed(19); np.random.seed(19); torch.manual_seed(19)
    py=random.getstate(); numpy=np.random.get_state(); tensor=torch.get_rng_state().clone()
    a=StudyCamera(config(kind),0).apply(im,row)
    b=StudyCamera(config(kind),0).apply(im,row)
    np.testing.assert_array_equal(a[0],b[0])
    assert a[0].size==im.size and a[0].mode=='RGB' and a[1]
    assert random.getstate()==py
    assert all(np.array_equal(x,y) for x,y in zip(np.random.get_state(),numpy))
    assert torch.equal(torch.get_rng_state(),tensor)
    assert not np.array_equal(a[0],im)


@pytest.mark.parametrize('kind',['none','white_balance','local_exposure','resolution','defocus','legacy'])
def test_zero_probability_is_exact_identity(kind):
    im=image(); result,applied=StudyCamera(config(kind,p=0),1).apply(im,dict(night_path='x',source_index=1))
    assert result is im and not applied


def test_v1_control_is_pixel_exact():
    profile=json.loads(Path('configs/night_camera_exploratory_v1.json').read_text())
    assert legacy_profile(.8,1.)==profile
    im=image()
    for epoch in range(6):
        row=dict(night_path='x',night_sha256='abc',source_index=1)
        a=StudyCamera(config('legacy',p=.8),epoch).apply(im,row)[0]
        b=CameraAugmentation(profile).apply(im,seed=camera_seed(2026,epoch,'abc',1),condition='night')
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('s',[.5,1.,1.5])
def test_scaled_legacy_stays_valid(s):
    CameraAugmentation(legacy_profile(.8,s))


@pytest.mark.parametrize('kind',['white_balance','local_exposure','resolution','defocus','legacy'])
def test_night_only_hook_preserves_replay_and_labels(replay_inputs,kind):
    rows,all_rows,c,labels=replay_inputs
    clean=night_train.PairedReplay(rows,all_rows,c,0)
    augmented=night_train.PairedReplay(rows,all_rows,dict(c,augmentation_study=config(kind)['augmentation_study']),0)
    np.testing.assert_array_equal(clean.replay,augmented.replay)
    np.testing.assert_array_equal(clean.order,augmented.order)
    for i in range(len(clean)):
        a,b=clean[i],augmented[i]
        assert a[2:]==b[2:6]
        torch.testing.assert_close(a[1],b[1],rtol=0,atol=0)


def test_profile_rejects_abandoned_and_invalid_values():
    for change in [dict(kind='fog'),dict(kind='water_drops'),dict(kind='glare'),dict(probability=1.1),dict(severity=float('nan'))]:
        with pytest.raises(ValueError):
            validate_profile(dict(config('none')['augmentation_study'],**change))


def test_controlled_config_and_immutable_output():
    c=json.loads(Path('configs/boq_b2_night_v1.json').read_text())
    c.update(output_dir='runs/augmentation_screen_v1/test/train', epochs=1,
             total_optimizer_steps=908,augmentation_study=config('none')['augmentation_study'])
    assert study_identity(c)['teacher']=='clean'
    for change in [dict(init_checkpoint='other.pt'),dict(gate_cos_min=.1),dict(output_dir='runs/boq_b2_night_v1/train')]:
        with pytest.raises(ValueError):study_identity(dict(c,**change))
