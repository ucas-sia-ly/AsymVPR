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
