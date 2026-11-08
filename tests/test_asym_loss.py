import numpy as np
import torch
from torch.nn import functional as F
from reproduction.asym_model import asym_loss
from reproduction.data import place_statistics


def test_eq8_value_and_gradient_match_direct_formula():
    torch.manual_seed(5)
    q = F.normalize(torch.randn(3, 8, dtype=torch.float64), dim=-1).requires_grad_()
    g = F.normalize(torch.randn(3, 8, dtype=torch.float64), dim=-1)
    means = torch.randn(7, 8, dtype=torch.float64) / 10
    var = torch.rand(7, 8, dtype=torch.float64) / 100
    ids = torch.tensor([0, 3, 6])
    actual = asym_loss(q, g, ids, means, var, chunk_size=2)
    values = []
    for i in range(3):
        pos = torch.exp(q[i] @ g[i] / .05)
        neg = sum(torch.exp(q[i] @ means[j] / .05 + 15 / (2 * .05**2) *
                           (q[i].square() * var[ids[i]]).sum()) for j in range(7) if j != ids[i])
        values.append(-torch.log(pos / (pos + neg)))
    expected = torch.stack(values).mean()
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, q, retain_graph=True)[0],
                               torch.autograd.grad(expected, q)[0])


def test_gamma_zero_and_own_place_exclusion():
    q = torch.tensor([[1., 0.]])
    g = q.clone()
    means = torch.tensor([[1000., 1000.], [0., 1.]])
    loss = asym_loss(q, g, torch.tensor([0]), means, torch.ones_like(means), gamma=0)
    torch.testing.assert_close(loss, F.softplus(torch.tensor(-20.)))


def test_statistics_use_all_images_and_do_not_normalize_means(tmp_path):
    x = np.array([[1., 0.], [0., 1.], [0., -1.]], dtype='float32')
    place_statistics(x, np.array([0, 2, 3]), tmp_path)
    np.testing.assert_allclose(np.load(tmp_path / 'centroids.npy'), [[.5, .5], [0., -1.]])
    np.testing.assert_allclose(np.load(tmp_path / 'variances.npy'), [[.5, .5], [0., 0.]])
