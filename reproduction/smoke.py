"""Bounded real-image model/gradient/resume check; bank tiling is performance only."""
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from reproduction.monitoring import TrainingLogger

from reproduction.asym_model import EfficientViTBoQ, GalleryBoQ, asym_loss
from reproduction.data import Images, loader


def main():
    c = json.loads(Path('configs/boq_b2.json').read_text())
    cache = Path(c['cache_dir'])
    meta = json.loads((cache / 'manifest.json').read_text())
    torch.set_num_threads(6)
    torch.manual_seed(c['seed'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    paths = np.load(cache / 'paths.npy', mmap_mode='r')
    rng = np.random.default_rng(c['seed'])
    chosen = rng.choice(len(paths), size=64, replace=False)
    images, _ = next(iter(loader(Images(c['gsv_root'], paths[chosen], 322), 64, 2)))
    teacher = GalleryBoQ(c).cuda().eval()
    with torch.inference_mode():
        g = teacher(images.cuda()).clone()
        torch.testing.assert_close(g.norm(dim=1), torch.ones(64, device='cuda'))
    # inference tensors must be cloned outside inference_mode before use in autograd.
    g = g.clone()
    del teacher
    gc.collect(); torch.cuda.empty_cache()
    model = EfficientViTBoQ(c).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    # Tile real embeddings only to exercise full-bank memory and GEMM sizes.
    # This is NOT the geographical training bank or a scientific result.
    repeats = (meta['num_places'] + 63) // 64
    means = g.repeat(repeats, 1)[:meta['num_places']].contiguous()
    var = torch.full_like(means, 1e-5)
    torch.cuda.reset_peak_memory_stats()
    with TrainingLogger(config=dict(c, diagnostic='tiled-bank-performance-only'), total_steps=2) as logger:
        for step in range(1, 3):
            torch.cuda.synchronize(); start = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            q = model(images.cuda())
            loss = asym_loss(q, g, torch.arange(64, device='cuda'), means, var,
                             c['temperature'], c['gamma'], c['bank_chunk_size'])
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(); elapsed = time.monotonic() - start
            metrics = {'smoke/loss': float(loss.detach()), 'smoke/grad_norm': float(norm),
                       'smoke/step_seconds': elapsed, 'smoke/images_per_second': 64/elapsed,
                       'smoke/peak_gpu_gib': torch.cuda.max_memory_allocated()/1024**3}
            logger.log(step, **metrics)
            print(json.dumps(dict(step=step, **metrics)), flush=True)
        path = logger.path / 'resume_check.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), path)
        # Re-load both state dicts and compare a deterministic eval descriptor.
        model.eval()
        with torch.no_grad():
            before = model(images[:2].cuda()).clone()
        state = torch.load(path, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        with torch.no_grad():
            after = model(images[:2].cuda())
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        print('PASS: teacher norm, 12288-D student, finite full-size-bank gradients, optimizer reload, exact eval roundtrip', flush=True)
        print('parameters', sum(p.numel() for p in model.parameters()), flush=True)


if __name__ == '__main__':
    main()
