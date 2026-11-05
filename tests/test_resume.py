import random
import numpy as np
import torch
from reproduction.run import save_checkpoint


def test_checkpoint_restores_optimizer_scheduler_rng_and_next_update(tmp_path):
    torch.manual_seed(42)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=.001)
    x, y = torch.randn(4, 3), torch.randn(4, 2)
    def update(m, opt):
        opt.zero_grad()
        (m(x) - y).square().mean().backward()
        opt.step()
    update(model, optimizer)
    scheduler.step()
    # CUDA state access is supplied by production; this CPU unit test uses no CUDA allocation.
    from unittest.mock import patch
    with patch('torch.cuda.get_rng_state_all', return_value=[]):
        save_checkpoint(tmp_path / 'last.pt', model, optimizer, scheduler, 1, 3, 4, 5., {}, {})
    expected_random = (random.random(), float(np.random.rand()), torch.rand(2))
    update(model, optimizer)
    saved = torch.load(tmp_path / 'last.pt', weights_only=False)
    restored = torch.nn.Linear(3, 2)
    restored.load_state_dict(saved['model'])
    opt2 = torch.optim.AdamW(restored.parameters(), lr=.01)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, 15, eta_min=.001)
    opt2.load_state_dict(saved['optimizer']); sch2.load_state_dict(saved['scheduler'])
    random.setstate(saved['random_state']); np.random.set_state(saved['numpy_state'])
    torch.set_rng_state(saved['torch_state'])
    assert random.random() == expected_random[0]
    assert float(np.random.rand()) == expected_random[1]
    torch.testing.assert_close(torch.rand(2), expected_random[2], rtol=0, atol=0)
    update(restored, opt2)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert scheduler.state_dict() == sch2.state_dict()
    assert (saved['epoch'], saved['next_batch'], saved['step']) == (1, 3, 4)
