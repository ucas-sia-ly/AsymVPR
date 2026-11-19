"""Content-gated saturation and defocus on the final, unnormalized model input."""
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch

from reproduction.camera_augmentation import camera_seed


def as_image(tensor):
    return Image.fromarray(np.rint(tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def statistics(image):
    rgb = np.asarray(image, dtype=np.float32) / 255
    y = rgb @ np.array([.2126, .7152, .0722], np.float32)
    hi, lo = rgb.max(axis=2), rgb.min(axis=2)
    lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4*y[1:-1, 1:-1]
    return float(((hi-lo)/np.maximum(hi, 1e-5)).mean()), float(lap.std()), float(
        (y < .05).mean()), float((hi >= 254/255).mean())


def gradients(image):
    y = np.asarray(image.convert('L').filter(ImageFilter.GaussianBlur(.6)), dtype=np.float32)
    return np.stack([y[1:-1, 2:] - y[1:-1, :-2], y[2:, 1:-1] - y[:-2, 1:-1]], axis=-1)


def degrade(tensor, row, seed, epoch, edge_cutoff):
    rng = np.random.default_rng(camera_seed('gsv_night_v2', seed, epoch,
                                            row['night_sha256'], row['source_index']))
    branch = rng.random()
    if branch < .35:
        return tensor, (False, False, False)
    moderate = branch >= .85
    image = as_image(tensor)
    saturation, edge, dark, clip = statistics(image)
    eligible = []
    if saturation >= (.50 if moderate else .35):
        eligible.append('saturation')
    if edge >= edge_cutoff:
        eligible.append('defocus')
    if not eligible:
        return tensor, (False, False, False)
    count = min(len(eligible), 2 if moderate or rng.random() < .35 else 1)
    weights = np.array([5 if op == 'saturation' else 3 for op in eligible], dtype=float)
    before = gradients(image)
    energy = np.sum(before**2, axis=-1)
    mask = energy >= np.quantile(energy, .75)
    a = before[mask].ravel()
    for _ in range(3):
        ops = rng.choice(eligible, size=count, replace=False, p=weights/weights.sum())
        sat, blur = 'saturation' in ops, 'defocus' in ops
        result = image
        if sat:
            factor = rng.uniform(*((.65, .80) if moderate else (.80, .95)))
            result = ImageEnhance.Color(result).enhance(float(factor))
        if blur:
            sigma = rng.uniform(*((.55, .80) if moderate else (.30, .55)))
            result = result.filter(ImageFilter.GaussianBlur(float(sigma)))
        _, _, after_dark, after_clip = statistics(result)
        b = gradients(result)[mask].ravel()
        cosine = np.dot(a, b) / max(1e-10, np.linalg.norm(a) * np.linalg.norm(b))
        if (after_dark-dark <= .10 and after_clip-clip <= .005
                and cosine >= (.90 if moderate else .95)):
            if result.tobytes() == image.tobytes():
                return tensor, (False, False, False)
            pixels = torch.from_numpy(np.array(result)).permute(2, 0, 1).float().div_(255)
            return pixels, (sat, blur, False)
    return tensor, (False, False, True)
