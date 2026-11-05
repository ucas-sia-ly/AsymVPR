"""Synthetic CPU checks of night targets, branch gradients and loss weighting."""
import pytest
import torch
from torch.nn import functional as F

from reproduction.asym_model import asym_loss
from reproduction.night_objective import mix_night_target, night_objective


def teachers_at_cosines(cosines):
    cosines = torch.tensor(cosines, dtype=torch.float64)
    day = torch.zeros((len(cosines), 2), dtype=torch.float64)
    day[:, 0] = 1.
    night = torch.stack([cosines, (1. - cosines.square()).clamp_min(0).sqrt()], dim=-1)
    return night, day


def test_teacher_agreement_gate_has_expected_thresholds_and_unit_targets():
    night, day = teachers_at_cosines([-1., 0., .2, .45, .7, 1.])
    target, weight, agreement = mix_night_target(night, day)
    torch.testing.assert_close(agreement, torch.tensor([-1., 0., .2, .45, .7, 1.], dtype=torch.float64))
    torch.testing.assert_close(weight, torch.tensor([0., 0., 0., .25, .5, .5], dtype=torch.float64))
    torch.testing.assert_close(target.norm(dim=1), torch.ones(6, dtype=torch.float64))
    torch.testing.assert_close(target[:3], day[:3])
    torch.testing.assert_close(target[4], F.normalize(day[4] + night[4], dim=0))


def test_opposite_teachers_keep_day_anchor_without_cancellation():
    night, day = teachers_at_cosines([-1.])
    target, weight, _ = mix_night_target(night, day)
    torch.testing.assert_close(target, day)
    assert weight.item() == 0.
    assert torch.isfinite(target).all()


def test_target_and_gate_stop_teacher_gradients():
    night, day = teachers_at_cosines([.45, .8])
    night.requires_grad_()
    day.requires_grad_()
    target, weight, agreement = mix_night_target(night, day)
    for value in (target, weight, agreement):
        assert not value.requires_grad
        assert value.grad_fn is None


def objective_fixture():
    generator = torch.Generator().manual_seed(19)
    q_night = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    q_replay = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    day = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    night = F.normalize(day.detach() + .3 * torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    replay = F.normalize(torch.randn(3, 8, generator=generator, dtype=torch.float64), dim=-1).requires_grad_()
    kwargs = dict(teacher_night=night, teacher_source_day=day, teacher_replay_day=replay,
                  night_place_ids=torch.tensor([1, 4, 6]), replay_place_ids=torch.tensor([2, 4, 0]),
                  centroids=torch.randn(7, 8, generator=generator, dtype=torch.float64) / 10,
                  variances=torch.rand(7, 8, generator=generator, dtype=torch.float64) / 100,
                  chunk_size=2)
    return q_night, q_replay, kwargs


def test_balanced_loss_and_gradients_match_one_concatenated_eq8_batch():
    q_night, q_replay, kwargs = objective_fixture()
    actual, metrics = night_objective(q_night, q_replay, **kwargs)
    target, weight, agreement = mix_night_target(kwargs['teacher_night'], kwargs['teacher_source_day'])
    expected = asym_loss(torch.cat([q_night, q_replay]),
        torch.cat([target, kwargs['teacher_replay_day'].detach()]),
        torch.cat([kwargs['night_place_ids'], kwargs['replay_place_ids']]),
        kwargs['centroids'], kwargs['variances'], chunk_size=2)
    torch.testing.assert_close(actual, expected)
    actual_grad = torch.autograd.grad(actual, (q_night, q_replay), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (q_night, q_replay))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)
        assert torch.isfinite(left).all() and left.norm() > 0
    actual.backward()
    for name in ('teacher_night', 'teacher_source_day', 'teacher_replay_day'):
        assert kwargs[name].grad is None
    assert set(metrics) == {'night_asym', 'replay_asym', 'teacher_daynight_cos',
                            'night_teacher_weight', 'qnight_teacherday_cos', 'qnight_teachernight_cos'}
    for value in metrics.values():
        assert value.ndim == 0 and torch.isfinite(value)
        assert not value.requires_grad and value.grad_fn is None
    torch.testing.assert_close(metrics['teacher_daynight_cos'], agreement.mean())
    torch.testing.assert_close(metrics['night_teacher_weight'], weight.mean())
    torch.testing.assert_close(actual.detach(), .5 * (metrics['night_asym'] + metrics['replay_asym']))


def test_zero_night_weight_still_supervises_all_night_queries():
    q_night, q_replay, kwargs = objective_fixture()
    actual, _ = night_objective(q_night, q_replay, max_night_weight=0., **kwargs)
    expected_night = asym_loss(q_night, kwargs['teacher_source_day'].detach(),
        kwargs['night_place_ids'], kwargs['centroids'], kwargs['variances'], chunk_size=2)
    expected_replay = asym_loss(q_replay, kwargs['teacher_replay_day'].detach(),
        kwargs['replay_place_ids'], kwargs['centroids'], kwargs['variances'], chunk_size=2)
    torch.testing.assert_close(actual, .5 * (expected_night + expected_replay))
    gradient, = torch.autograd.grad(actual, q_night)
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_one_bank_pass_matches_two_branches_for_unequal_batch_sizes():
    q_night, full_replay, kwargs = objective_fixture()
    q_replay = full_replay[:2]
    kwargs['teacher_replay_day'] = kwargs['teacher_replay_day'][:2]
    kwargs['replay_place_ids'] = kwargs['replay_place_ids'][:2]
    actual, metrics = night_objective(q_night, q_replay, **kwargs)
    target, _, _ = mix_night_target(kwargs['teacher_night'], kwargs['teacher_source_day'])
    direct_night = asym_loss(q_night, target, kwargs['night_place_ids'],
                             kwargs['centroids'], kwargs['variances'], chunk_size=2)
    direct_replay = asym_loss(q_replay, kwargs['teacher_replay_day'].detach(),
        kwargs['replay_place_ids'], kwargs['centroids'], kwargs['variances'], chunk_size=2)
    expected = .5 * (direct_night + direct_replay)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(metrics['night_asym'], direct_night.detach())
    torch.testing.assert_close(metrics['replay_asym'], direct_replay.detach())
    actual_grad = torch.autograd.grad(actual, (q_night, q_replay), retain_graph=True)
    expected_grad = torch.autograd.grad(expected, (q_night, q_replay))
    for left, right in zip(actual_grad, expected_grad):
        torch.testing.assert_close(left, right)


def test_eq8_none_reduction_preserves_default_mean_and_gradient():
    q_night, _, kwargs = objective_fixture()
    per_query = asym_loss(q_night, kwargs['teacher_source_day'].detach(),
        kwargs['night_place_ids'], kwargs['centroids'], kwargs['variances'],
        chunk_size=2, reduction='none')
    default = asym_loss(q_night, kwargs['teacher_source_day'].detach(),
        kwargs['night_place_ids'], kwargs['centroids'], kwargs['variances'], chunk_size=2)
    assert per_query.shape == (len(q_night),)
    torch.testing.assert_close(default, per_query.mean())
    expected_grad, = torch.autograd.grad(default, q_night)
    actual_grad, = torch.autograd.grad(per_query.mean(), q_night)
    torch.testing.assert_close(actual_grad, expected_grad)
    with pytest.raises(ValueError, match='reduction'):
        asym_loss(q_night, kwargs['teacher_source_day'].detach(), kwargs['night_place_ids'],
                  kwargs['centroids'], kwargs['variances'], reduction='sum')


@pytest.mark.parametrize('override', [dict(gate_cos_span=0.), dict(max_night_weight=.6),
                                     dict(gate_cos_min=float('nan'))])
def test_invalid_gate_configuration_is_rejected(override):
    night, day = teachers_at_cosines([.5])
    with pytest.raises(ValueError, match='gate parameters'):
        mix_night_target(night, day, **override)


def test_mismatched_teacher_batch_is_rejected():
    q_night, q_replay, kwargs = objective_fixture()
    kwargs['teacher_source_day'] = kwargs['teacher_source_day'][:2]
    with pytest.raises(ValueError, match='shapes'):
        night_objective(q_night, q_replay, **kwargs)
