"""Replay the legacy camera draws with one operation removed."""
import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from reproduction.camera_augmentation import (
    CameraAugmentation, _directional_blur, _linear_to_srgb, _pil, _srgb_to_linear,
)


class LegacyCameraAblation(CameraAugmentation):
    # The original profile first runs with tracing enabled. The second pass
    # starts each retained op at its original RNG state, even after Poisson noise.
    def apply(self, image, *, seed: int, condition: str, omit=None):
        if omit is None:
            return super().apply(image, seed=seed, condition=condition)
        if omit not in self.profile["operations"]:
            raise ValueError(f"Unknown camera omission: {omit}")
        trace = {}
        self._apply(image, seed, condition, trace=trace)
        return self._apply(image, seed, condition, omit=omit, trace=trace)

    def _apply(self, image, seed, condition, omit=None, trace=None):
        if condition not in self.profile["conditions"] or self.profile.get("probability", 1.) == 0:
            return image
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Camera seed must be a nonnegative integer")
        generator = np.random.default_rng(seed)
        if generator.random() >= self.profile.get("probability", 1.):
            return image
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.

        def enabled(name):
            op = self.profile["operations"].get(name)
            if omit is not None:
                if name == omit or name not in trace:
                    return None
                generator.bit_generator.state = trace[name]
                return op
            if op is not None and generator.random() < op.get("probability", 1.):
                if trace is not None:
                    trace[name] = generator.bit_generator.state
                return op
            return None

        def draw(op, name, *, integer=False):
            low, high = op[name]
            return int(generator.integers(low, high + 1)) if integer else float(generator.uniform(low, high))

        if (op := enabled("gain_gamma")) is not None:
            # Deliberate response simulation in normalized display values;
            # neither scene radiance nor the camera auto-exposure loop is known.
            x = np.clip(x * draw(op, "gain"), 0, 1) ** draw(op, "gamma")
        if (op := enabled("saturation")) is not None:
            x = np.asarray(ImageEnhance.Color(_pil(x)).enhance(draw(op, "factor")), dtype=np.float32) / 255.
        if (op := enabled("bloom")) is not None:
            threshold = draw(op, "threshold")
            highlight = np.maximum(x - threshold, 0) / max(1e-8, 1 - threshold)
            glow = np.asarray(_pil(highlight).filter(ImageFilter.GaussianBlur(draw(op, "radius"))),
                              dtype=np.float32) / 255.
            x = np.clip(x + draw(op, "strength") * glow, 0, 1)
        if (op := enabled("motion_blur")) is not None:
            x = _directional_blur(x, draw(op, "length", integer=True), draw(op, "angle"))
        if (op := enabled("vignette")) is not None:
            yy, xx = np.meshgrid(np.linspace(-1., 1., x.shape[0]), np.linspace(-1., 1., x.shape[1]), indexing="ij")
            x = x * (1 - draw(op, "strength") * (xx ** 2 + yy ** 2) / 2)[..., None]
        if (op := enabled("noise")) is not None:
            peak = draw(op, "shot_peak")
            linear = _srgb_to_linear(np.clip(x, 0, 1))
            noisy = generator.poisson(linear * peak) / peak
            noisy += generator.normal(0., draw(op, "read_std"), x.shape)
            x = _linear_to_srgb(np.clip(noisy, 0, 1))
        result = _pil(x)
        if (op := enabled("jpeg")) is not None:
            buffer = io.BytesIO()
            result.save(buffer, format="JPEG", quality=draw(op, "quality", integer=True))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                result = decoded.convert("RGB").copy()
        return result
