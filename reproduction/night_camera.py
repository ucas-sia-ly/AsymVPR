"""Seeded WORKSPACE camera perturbations, applied to original night pixels."""
from pathlib import Path

from reproduction.camera_augmentation import CameraAugmentation, camera_seed
from reproduction.evaluate import sha256

ORDER = 'before_to_tensor_resize_normalize'


def camera_identity(config):
    if not config.get('camera_profile'):
        return None
    if config.get('camera_order') != ORDER or config.get('camera_teacher_target') != 'clean_night':
        raise ValueError('Camera experiment requires pre-Tensor/Resize/Normalize order and clean-night teacher targets')
    profile = Path(config['camera_profile'])
    implementation = Path(__file__).with_name('camera_augmentation.py')
    if sha256(profile) != config.get('camera_profile_sha256'):
        raise ValueError('Camera profile SHA256 mismatch')
    if sha256(implementation) != config.get('camera_implementation_sha256'):
        raise ValueError('Camera implementation SHA256 mismatch')
    aug = CameraAugmentation.from_json(profile)
    return dict(profile=str(profile), profile_sha256=aug.sha256, profile_content=aug.profile,
        implementation_sha256=sha256(implementation),
        source='/home/admin123/github/WORKSPACE/salad/workflow/camera_augmentation.py',
        order=['original_size_RGB', 'camera_degradation', 'ToTensor', 'Resize_bicubic_antialias', 'ImageNet_Normalize'],
        parameter_pixel_units='original image pixels, before resizing',
        teacher_target='frozen clean generated-night features; never degraded',
        evaluation='clean original queries and fixed teacher galleries; never degraded',
        rng='local NumPy generator keyed by training seed, epoch and immutable night SHA/source index')


class NightCamera:
    def __init__(self, config, epoch):
        self.identity = camera_identity(config)
        self.augmentation = CameraAugmentation(self.identity['profile_content'])
        self.seed, self.epoch = config['seed'], epoch

    def apply(self, image, row):
        seed = camera_seed(self.seed, self.epoch, row.get('night_sha256', row['night_path']), row['source_index'])
        result = self.augmentation.apply(image, seed=seed, condition='night')
        # Upstream returns the input object exactly when the profile is skipped.
        return result, result is not image
