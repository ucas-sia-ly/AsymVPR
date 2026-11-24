"""The original and adjusted camera policies for the GSV final release."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from reproduction.camera_augmentation import (
    CameraAugmentation, camera_seed, _linear_to_srgb, _pil, _srgb_to_linear,
)
from reproduction.evaluate import sha256

LEGACY = Path('configs/night_camera_exploratory_v1.json')
OP_NAMES = ('saturation', 'white_balance', 'defocus', 'resolution', 'motion_blur',
            'noise', 'jpeg', 'gain_gamma', 'local_exposure', 'vignette', 'bloom')
PROCESSING_ORDER = ('gain_gamma', 'local_exposure', 'vignette', 'white_balance', 'saturation',
                    'defocus', 'resolution', 'motion_blur', 'noise', 'jpeg', 'bloom')
FAMILIES = (
    (.50, {'saturation': .75, 'white_balance': .25}),
    (.30, {'defocus': .45, 'resolution': .20, 'motion_blur': .20, 'noise': .10, 'jpeg': .05}),
    (.15, {'gain_gamma': .45, 'local_exposure': .40, 'vignette': .15}),
    (.05, {'bloom': 1.}),
)
# Pixel radii/lengths apply at model resolution. Gain and gamma may move in either direction.
RANGES = {
    'motion_blur': ({'length': [2, 3], 'angle': [-90, 90]}, {'length': [3, 5], 'angle': [-90, 90]}),
    'noise': ({'shot_peak': [1500, 4000], 'read_std': [0., .002]},
              {'shot_peak': [700, 1500], 'read_std': [0., .003]}),
    'jpeg': ({'quality': [90, 100]}, {'quality': [80, 90]}),
    'gain_gamma': ({'gain': [.94, 1.06], 'gamma': [.98, 1.02]},
                   {'gain': [.88, 1.10], 'gamma': [.96, 1.04]}),
    'vignette': ({'strength': [.02, .08]}, {'strength': [.08, .12]}),
    'bloom': ({'threshold': [.90, .98], 'strength': [.02, .08], 'radius': [.5, 2.]},
              {'threshold': [.90, .98], 'strength': [.05, .12], 'radius': [.5, 2.]}),
}


def identity(config):
    mode = config['gsv_camera']
    if not config.get('night_release_root') or mode not in ('legacy', 'adjusted'):
        raise ValueError('GSV camera policy requires the published dataset and a known mode')
    threshold = config.get('gsv_edge_threshold')
    if mode == 'adjusted' and (type(threshold) not in (int, float) or not 0 < threshold < 1):
        raise ValueError('Adjusted GSV camera requires a frozen training edge threshold')
    omitted = config.get('gsv_camera_omit', [])
    names = tuple(RANGES) + ('saturation',) if mode == 'legacy' else OP_NAMES
    if (not isinstance(omitted, list) or any(name not in names for name in omitted)
            or len(set(omitted)) != len(omitted) or (mode == 'legacy' and len(omitted) > 1)):
        raise ValueError('Invalid operations to omit from the GSV camera policy')
    return dict(mode=mode, edge_threshold=threshold, implementation_sha256=sha256(__file__),
                legacy_profile_sha256=sha256(LEGACY), operations=list(OP_NAMES) if mode == 'adjusted' else list(RANGES) + ['saturation'],
                omitted_operations=omitted,
                order='at model resolution before normalize' if mode == 'adjusted' else 'original RGB before resize',
                teacher_target='clean_night', student_branch='night_only')


def statistics(rgb):
    luma = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    high, low = rgb.max(axis=2), rgb.min(axis=2)
    lap = (luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2]
           + luma[1:-1, 2:] - 4 * luma[1:-1, 1:-1])
    return (float(luma.mean()), float(np.median(luma)), float((luma < .05).mean()),
            float((high >= 254/255).mean()), float(((high-low)/np.maximum(high, 1e-5)).mean()), float(lap.std()))


def gradients(image):
    y = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.6)), dtype=np.float32)
    return np.stack((y[1:-1, 2:] - y[1:-1, :-2], y[2:, 1:-1] - y[:-2, 1:-1]), axis=-1)


def choose(rng, weights):
    values = np.array(list(weights.values()), dtype=float)
    return rng.choice(list(weights), p=values/values.sum())


class GSVCamera:
    def __init__(self, config, epoch, transform):
        self.seed, self.epoch = config['seed'], epoch
        self.transform = transform
        self.resize_steps = transform.transforms[:-1]
        self.normalize = transform.transforms[-1]
        self.threshold = config.get('gsv_edge_threshold')
        self.omitted = frozenset(config.get('gsv_camera_omit', []))
        self.legacy = None
        if config['gsv_camera'] == 'legacy':
            from reproduction.legacy_camera_ablation import LegacyCameraAblation
            self.legacy = LegacyCameraAblation.from_json(LEGACY)
        self.ops = {name: [CameraAugmentation(dict(version=1, status='exploratory',
            conditions=['night'], operations={name: limits})) for limits in ranges]
            for name, ranges in RANGES.items()} if self.legacy is None else {}

    def operation(self, image, name, moderate, rng):
        if name in self.ops:
            return self.ops[name][moderate].apply(image, seed=int(rng.integers(2**63)), condition='night')
        if name == 'saturation':
            return ImageEnhance.Color(image).enhance(float(rng.uniform(*((.65,.80) if moderate else (.80,.95)))))
        if name == 'defocus':
            return image.filter(ImageFilter.GaussianBlur(float(rng.uniform(*((.55,.80) if moderate else (.30,.55))))))
        width, height = image.size
        if name == 'resolution':
            ratio = rng.uniform(*((.70,.85) if moderate else (.85,.98)))
            return image.resize((round(width*ratio), round(height*ratio)), Image.Resampling.BOX).resize(
                image.size, Image.Resampling.BICUBIC)
        linear = _srgb_to_linear(np.asarray(image, dtype=np.float32)/255)
        if name == 'white_balance':
            limit = .13 if moderate else .075
            gains = rng.uniform(-limit, limit, size=3)
            linear *= np.exp(gains - gains.mean()).astype(np.float32)
        else:
            ev = rng.uniform(-1, 1, size=(4,4)).astype(np.float32)
            field = np.array(Image.fromarray(ev).resize(image.size, Image.Resampling.BICUBIC))
            field -= field.mean()
            amplitude = rng.uniform(*((.35,.60) if moderate else (.15,.35)))
            field *= amplitude / max(float(np.ptp(field)), 1e-6)
            linear *= np.exp2(field[..., None])
        return _pil(_linear_to_srgb(np.clip(linear, 0, 1)))

    def apply(self, image, row):
        key = camera_seed(self.seed, self.epoch, row['night_sha256'], row['source_index'])
        flags = np.zeros(2 + len(OP_NAMES), dtype=np.uint8)
        if self.legacy is not None:
            result = self.legacy.apply(image, seed=key, condition='night', omit=next(iter(self.omitted), None))
            flags[0] = result is not image
            return self.transform(result), flags

        rng = np.random.default_rng(key)
        draw = rng.random()
        clean = image
        for step in self.resize_steps:
            clean = step(clean)
        if draw < .35:
            return self.normalize(clean), flags
        moderate = draw >= .85
        resized = _pil(clean.permute(1,2,0).numpy())
        baseline_rgb = np.asarray(resized, dtype=np.float32)/255
        mean, median, dark, clip, sat, edge = statistics(baseline_rgb)
        protected_dark = median < .07 or dark > .50
        excluded = set()
        if sat < (.50 if moderate else .35):
            excluded.add('saturation')
        if edge < self.threshold:
            excluded.update(('defocus', 'resolution', 'motion_blur'))
        if protected_dark:
            excluded.update(('vignette', 'noise', 'local_exposure'))
        if clip > .01 or not (baseline_rgb >= .90).any():
            excluded.add('bloom')
        families = [(w, {name: weight for name, weight in ops.items() if name not in excluded})
                    for w, ops in FAMILIES]
        weights = {i:w for i,(w,ops) in enumerate(families) if ops}
        if not weights:
            return self.normalize(clean), flags
        count = min(len(weights), 2 if moderate or rng.random() < .35 else 1)
        before = gradients(resized)
        energy = (before**2).sum(axis=2)
        mask = energy >= np.quantile(energy, .75)
        a = before[mask].ravel()
        norm_a = np.linalg.norm(a)

        def acceptable(rgb, result, selected):
            if np.array_equal(rgb, baseline_rgb):
                return False
            new_mean, _, new_dark, new_clip, _, _ = statistics(rgb)
            if (new_dark-dark > .10 or new_clip-clip > .005
                    or ('gain_gamma' in selected and ((protected_dark and new_mean < mean)
                        or (clip > .01 and new_mean > mean)))):
                return False
            b = gradients(result)[mask].ravel()
            cosine = np.dot(a,b) / max(1e-10, norm_a*np.linalg.norm(b))
            return cosine >= (.90 if moderate else .95)

        for _ in range(3):
            choices = weights.copy()
            selected = []
            for _ in range(count):
                family = choose(rng, choices)
                selected.append(choose(rng, families[family][1]))
                del choices[family]
            result = resized
            selected.sort(key=PROCESSING_ORDER.index)
            states = {}
            for operation in selected:
                if self.omitted:
                    states[operation] = rng.bit_generator.state
                result = self.operation(result, operation, moderate, rng)
            rgb = np.array(result, dtype=np.float32)/255
            if not acceptable(rgb, result, selected):
                continue
            if self.omitted.intersection(selected):
                selected = [name for name in selected if name not in self.omitted]
                if not selected:
                    return self.normalize(clean), flags
                result = resized
                for operation in selected:
                    rng.bit_generator.state = states[operation]
                    result = self.operation(result, operation, moderate, rng)
                rgb = np.array(result, dtype=np.float32)/255
                if not acceptable(rgb, result, selected):
                    flags[1] = 1
                    return self.normalize(clean), flags
            flags[0] = 1
            for operation in selected:
                flags[2+OP_NAMES.index(operation)] = 1
            return self.normalize(torch.from_numpy(rgb).permute(2,0,1)), flags
        flags[1] = 1
        return self.normalize(clean), flags
