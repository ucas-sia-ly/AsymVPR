"""Bounded FP32 inference-only extraction batch comparison."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from reproduction.monitoring import TrainingLogger
from reproduction.asym_model import GalleryBoQ
from reproduction.data import Images, loader

def main():
    c = json.loads(Path('configs/boq_b2.json').read_text())
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    model = GalleryBoQ(c).cuda().eval()
    paths = np.load(Path(c['cache_dir']) / 'paths.npy', mmap_mode='r')
    x, _ = next(iter(loader(Images(c['gsv_root'], paths[:64], 322), 64, 2)))
    x = x.cuda()
    with torch.inference_mode(), TrainingLogger(config=dict(c, diagnostic='gallery-inference-batch-comparison'), total_steps=3) as log:
        model(x)
        baseline = None
        for step, batch in enumerate((64, 128, 256), 1):
            inputs = x.repeat(batch // 64, 1, 1, 1)
            result = model(inputs)
            if baseline is None:
                baseline = result.clone()
            else:
                torch.testing.assert_close(result[:64], baseline, atol=1e-5, rtol=1e-4)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            for _ in range(2):
                result = model(inputs)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            metrics = dict(batch_size=batch, images_per_second=2*batch/elapsed,
                           peak_gpu_gib=torch.cuda.max_memory_allocated()/1024**3)
            log.log(step, **metrics)
            print(json.dumps(metrics), flush=True)
            del inputs, result

if __name__ == '__main__':
    main()
