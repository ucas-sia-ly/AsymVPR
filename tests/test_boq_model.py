"""Small CPU regressions; no pretrained downloads or training jobs."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.backbones import DinoV2
from src.boq import BoQ


def test_descriptor_only_preserves_checkpoint_keys_and_matches_attention_path():
    torch.manual_seed(12)
    model = BoQ(in_channels=8, proj_channels=64, num_queries=4, num_layers=2, row_dim=3).eval()
    keys = set(model.state_dict())
    with torch.inference_mode():
        inputs = torch.randn(2, 8, 3, 3)
        expected, maps = model(inputs)
        actual, omitted = model(inputs, return_attention=False)
    assert len(maps) == 2 and all(m.shape == (2, 4, 9) for m in maps)
    assert omitted == [] and set(model.state_dict()) == keys
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(actual.norm(dim=1), torch.ones(2))
    model.load_state_dict(model.state_dict(), strict=True)


class TinyDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_dim = 3
        self.patch_embed = SimpleNamespace(patch_size=(1, 1))
        self.blocks = nn.ModuleList([nn.Linear(3, 3) for _ in range(3)])
        self.calls = []
        for i, block in enumerate(self.blocks):
            block.register_forward_hook(lambda _m, _i, _o, i=i: self.calls.append(i))

    def prepare_tokens_with_masks(self, images):
        patches = images.flatten(2).transpose(1, 2)
        return torch.cat((patches[:, :1], patches), dim=1)


@pytest.mark.parametrize('unfreeze', [0, 1, 3])
def test_dino_injection_freezes_correct_blocks_and_runs_each_once(unfreeze, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Injected backbone must not download a model')
    monkeypatch.setattr(torch.hub, 'load', forbidden)
    dino = TinyDino()
    model = DinoV2(dino=dino, unfreeze_n_blocks=unfreeze)
    images = torch.randn(2, 3, 2, 2)
    with torch.no_grad():
        tokens = dino.prepare_tokens_with_masks(images)
        for block in dino.blocks:
            tokens = block(tokens)
        expected = tokens[:, 1:].transpose(1, 2).reshape(2, 3, 2, 2)
    dino.calls.clear()
    output = model(images)
    assert dino.calls == [0, 1, 2]
    torch.testing.assert_close(output, expected)
    assert output.requires_grad == (unfreeze > 0)
    if unfreeze:
        output.sum().backward()
    for i, block in enumerate(dino.blocks):
        for parameter in block.parameters():
            assert parameter.requires_grad == (i >= 3 - unfreeze)
            assert (parameter.grad is not None) == (i >= 3 - unfreeze)


@pytest.mark.parametrize('unfreeze', [-1, 4, True])
def test_dino_rejects_invalid_unfreeze_count(unfreeze):
    with pytest.raises(ValueError, match='unfreeze_n_blocks'):
        DinoV2(dino=TinyDino(), unfreeze_n_blocks=unfreeze)
