"""Bounded inference benchmark; synthetic input timings are not retrieval scores."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ
from reproduction.data import atomic_json
from reproduction.monitoring import TrainingLogger
from reproduction.evaluate import load_inference_checkpoint, validate_inference_checkpoint


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/boq_b2.json')
    parser.add_argument('--checkpoint', default='checkpoints/boq_b2_best.pt')
    parser.add_argument('--output', default='results/efficiency.json')
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--iterations', type=int, default=50)
    args = parser.parse_args()
    if args.warmup < 1 or args.iterations < 2:
        parser.error('Use at least one warmup and two measured iterations')
    c = json.loads(Path(args.config).read_text())
    teacher_sha = digest(c['teacher_weights'])
    checkpoint_state = load_inference_checkpoint(args.checkpoint)
    validate_inference_checkpoint(checkpoint_state, c, teacher_sha)
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(2024)
    x = torch.randn(1, 3, c['image_size'], c['image_size'], device='cuda')
    result = {'torch': torch.__version__, 'python': platform.python_version(),
              'gpu': torch.cuda.get_device_name(), 'precision': 'float32', 'tf32': False,
              'batch_size': 1, 'image_size': c['image_size'], 'warmup': args.warmup,
              'iterations': args.iterations, 'checkpoint_sha256': digest(args.checkpoint),
              'teacher_sha256': teacher_sha, 'config_sha256': digest(args.config),
              'timing_protocol': 'Synchronized wall clock model forward on a reused synthetic CUDA input; preprocessing, transfers, retrieval and model loading excluded. Desktop GPU, not edge hardware.',
              'models': {}}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with TrainingLogger(config={'task': 'inference_efficiency', **vars(args)}, total_steps=2) as logger:
        for step, name in enumerate(('gallery', 'query'), 1):
            if name == 'gallery':
                model = GalleryBoQ(c)
            else:
                model = EfficientViTBoQ(c, pretrained=False)
                model.load_state_dict(checkpoint_state['model'], strict=True)
                del checkpoint_state
            params = sum(p.numel() for p in model.parameters())
            model = model.cuda().eval()
            timings = []
            with torch.inference_mode():
                for _ in range(args.warmup):
                    embedding = model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(args.iterations):
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    embedding = model(x)
                    torch.cuda.synchronize()
                    timings.append(1000 * (time.perf_counter() - start))
                assert embedding.shape == (1, c['descriptor_dim'])
                if not torch.isfinite(embedding).all():
                    raise FloatingPointError('Nonfinite descriptor')
                torch.testing.assert_close(embedding.norm(dim=1), torch.ones(1, device='cuda'))
            record = {'parameters': params, 'latency_ms_median': statistics.median(timings),
                      'latency_ms_mean': statistics.mean(timings),
                      'latency_ms_min': min(timings), 'latency_ms_max': max(timings),
                      'latency_ms_samples': timings,
                      'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3}
            result['models'][name] = record
            logger.log(step, **{f'efficiency/{name}/{k}': v for k, v in record.items() if isinstance(v, (int, float))})
            print(json.dumps({'model': name, **{k: v for k, v in record.items() if k != 'latency_ms_samples'}}), flush=True)
            del model, embedding
            gc.collect()
            torch.cuda.empty_cache()
        result['query_speedup_median'] = result['models']['gallery']['latency_ms_median'] / result['models']['query']['latency_ms_median']
        atomic_json(args.output, result)
    print(f'Inference benchmark saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
