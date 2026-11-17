"""Isolated augmentation screening; legacy camera implementation stays immutable.

No weather/glare additions, no adaptive thresholds and no final-test selection.
"""
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from reproduction.camera_augmentation import (CameraAugmentation, camera_seed,
    _pil, _srgb_to_linear, _linear_to_srgb)
from reproduction.evaluate import sha256

KINDS = {'none', 'legacy', 'white_balance', 'local_exposure', 'resolution', 'defocus'}


def validate_profile(profile):
    if (set(profile) != {'version', 'kind', 'probability', 'severity'}
            or type(profile['version']) is not int or profile['version'] != 1):
        raise ValueError('Invalid augmentation-study profile schema')
    if profile['kind'] not in KINDS:
        raise ValueError('Unknown study operation (weather/glare additions excluded)')
    for key, low, high in [('probability', 0, 1), ('severity', .1, 1.5)]:
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f'Invalid study {key}')


def legacy_profile(probability, severity):
    """Scale distance from identity; s=1 reproduces the original seven-op profile."""
    profile = json.loads(Path('configs/night_camera_exploratory_v1.json').read_text())
    profile['probability'] = probability
    ops = profile['operations']
    for name, key, neutral in [('gain_gamma', 'gain', 1), ('gain_gamma', 'gamma', 1),
                               ('saturation', 'factor', 1), ('jpeg', 'quality', 100)]:
        values = [neutral + severity * (x-neutral) for x in ops[name][key]]
        ops[name][key] = [int(round(x)) for x in values] if key == 'quality' else values
    for name in ['bloom', 'vignette']:
        ops[name]['strength'] = [v*severity for v in ops[name]['strength']]
    ops['motion_blur']['length'] = [max(1, round(1+severity*(v-1))) for v in ops['motion_blur']['length']]
    ops['noise']['shot_peak'] = [v/severity**2 for v in ops['noise']['shot_peak']]
    ops['noise']['read_std'] = [v*severity for v in ops['noise']['read_std']]
    # Bloom threshold/radius and blur angle stay fixed: severity changes amplitude.
    return profile


def study_identity(config):
    if config.get('camera_profile'):
        raise ValueError('Study and legacy camera hooks cannot both be active')
    profile = config['augmentation_study']
    validate_profile(profile)
    base = json.loads(Path('configs/boq_b2_night_v1.json').read_text())
    # Only output paths, resource settings, explicit pilot budget and augmentation may differ.
    allowed = {'output_dir', 'experiment', 'epochs', 'total_optimizer_steps',
               'train_num_workers', 'num_workers', 'snapshot_workers', 'augmentation_study',
               'input_dir', 'night_teacher_cache', 'replay_excluded_place_ids', 'study_input_summary_sha256'}
    for key in set(base) | set(config):
        if key not in allowed and base.get(key) != config.get(key):
            raise ValueError(f'Study changes a controlled baseline field: {key}')
    if (config['input_dir'], config['night_teacher_cache']) != (base['input_dir'], base['night_teacher_cache']):
        root = Path('runs/augmentation_screen_remaining_v2')
        if (Path(config['input_dir']).resolve() != (root/'inputs').resolve()
                or Path(config['night_teacher_cache']).resolve() != (root/'cache/night_teacher.npy').resolve()):
            raise ValueError('Study must use the fixed remaining snapshot/cache pair')
        summary_path = root/'inputs/summary.json'
        if sha256(summary_path) != config.get('study_input_summary_sha256'):
            raise ValueError('Study remaining snapshot summary checksum mismatch')
        summary = json.loads(summary_path.read_text())
        if config.get('replay_excluded_place_ids') != summary['derivation']['replay_excluded_place_ids']:
            raise ValueError('Study must preserve all original validation-place replay exclusions')
    elif config.get('replay_excluded_place_ids') or config.get('study_input_summary_sha256'):
        raise ValueError('Subset controls require the remaining snapshot')
    output = Path(config['output_dir']).resolve()
    if not any(output.is_relative_to(Path(root).resolve()) for root in
               ['runs/augmentation_screen_v1','runs/augmentation_screen_remaining_v2']):
        raise ValueError('Study output must be separate from retained experiments')
    if type(config['epochs']) is not int or not 1 <= config['epochs'] <= 10:
        raise ValueError('Study budget must be 1–10 epochs')
    return dict(profile=copy.deepcopy(profile), implementation_sha256=sha256(__file__),
        legacy_implementation_sha256=sha256('reproduction/camera_augmentation.py'),
        legacy_profile_sha256=sha256('configs/night_camera_exploratory_v1.json'),
        order='original_RGB -> study operation -> unchanged ToTensor/Resize/Normalize',
        student_branch='generated night only', teacher='clean', day_replay='clean',
        selection='development only; never automatic final testing',
        severity_units='new blur/resolution relative to final input; legacy uses original pixels')


class StudyCamera:
    def __init__(self, config, epoch):
        self.profile = config['augmentation_study']
        validate_profile(self.profile)
        self.seed, self.epoch, self.size = config['seed'], epoch, config['image_size']
        self.legacy = (CameraAugmentation(legacy_profile(self.profile['probability'], self.profile['severity']))
                       if self.profile['kind'] == 'legacy' else None)

    def apply(self, image, row):
        # Same key as v1 makes legacy s=1, p=.8 an exact camera control.
        seed = camera_seed(self.seed, self.epoch, row.get('night_sha256', row['night_path']), row['source_index'])
        if self.legacy is not None:
            result = self.legacy.apply(image, seed=seed, condition='night')
            return result, result is not image
        rng = np.random.default_rng(seed)
        kind, s, p = (self.profile[k] for k in ('kind', 'severity', 'probability'))
        if kind == 'none' or rng.random() >= p:
            return image, False
        x = np.asarray(image.convert('RGB'), dtype=np.float32)/255.
        h, w = x.shape[:2]
        if kind == 'white_balance':
            gains = np.exp(rng.uniform(-.14*s, .14*s, 3))
            gains /= np.prod(gains)**(1/3)
            x = _linear_to_srgb(np.clip(_srgb_to_linear(x)*gains, 0, 1))
            result = _pil(x)
        elif kind == 'local_exposure':
            # Continuous low-frequency EV field; not a hard mask or geometric warp.
            grid = rng.uniform(-.5*s, .25*s, (4, 4)).astype(np.float32)
            field = Image.fromarray(grid).resize((w, h), Image.Resampling.BICUBIC)
            ev = np.clip(np.asarray(field), -.5*s, .25*s)
            result = _pil(_linear_to_srgb(np.clip(_srgb_to_linear(x)*np.exp2(ev[..., None]), 0, 1)))
        elif kind == 'resolution':
            factor = rng.uniform(max(.25, 1-.4*s), 1.)
            # Effective detail measured against network size, not high-res source.
            width = max(1, min(w, round(self.size*factor)))
            height = max(1, min(h, round(self.size*factor)))
            result = image.resize((width, height), Image.Resampling.BOX).resize((w, h), Image.Resampling.BICUBIC)
        elif kind == 'defocus':
            sigma = rng.uniform(.2*s, .8*s)
            # Final resize can be anisotropic: this is an approximate equal-area scale.
            radius = sigma*np.sqrt((w/self.size)*(h/self.size))
            result = image.filter(ImageFilter.GaussianBlur(float(radius)))
        else:
            raise ValueError(kind)
        return result, True
