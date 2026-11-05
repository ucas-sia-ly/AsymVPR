"""Teacher-consistency targets for night adaptation in the fixed gallery space."""
import math

import torch
from torch.nn import functional as F

from reproduction.asym_model import asym_loss


def mix_night_target(teacher_night, teacher_source_day, *, gate_cos_min=.2,
                     gate_cos_span=.5, max_night_weight=.5):
    """Return detached unit targets, night weights and teacher cosine agreement.

    The agreement is a soft weighting signal, not an image-quality label. With
    the default gate, agreement <= .2 uses only the exact source-day teacher;
    agreement >= .7 mixes the night/day teachers equally. Every night image
    retains its source-day anchor, including images with low teacher agreement.
    Inputs must be finite, nonzero teacher descriptors from verified caches.
    """
    if (teacher_night.ndim != 2 or teacher_night.shape != teacher_source_day.shape
            or not teacher_night.shape[0]):
        raise ValueError('Night and source-day teachers must have matching nonempty B x D shapes')
    if (not all(math.isfinite(value) for value in (gate_cos_min, gate_cos_span, max_night_weight))
            or not -1 <= gate_cos_min < 1 or gate_cos_span <= 0
            or not 0 <= max_night_weight <= .5):
        raise ValueError('Invalid teacher-consistency gate parameters')
    with torch.no_grad():
        day = F.normalize(teacher_source_day.detach(), dim=-1)
        night = F.normalize(teacher_night.detach(), dim=-1)
        agreement = (day * night).sum(-1).clamp(-1., 1.)
        weight = max_night_weight * ((agreement - gate_cos_min) / gate_cos_span).clamp(0., 1.)
        target = F.normalize((1. - weight[:, None]) * day + weight[:, None] * night, dim=-1)
    return target, weight, agreement


def night_objective(q_night, q_replay, *, teacher_night, teacher_source_day,
                    teacher_replay_day, night_place_ids, replay_place_ids,
                    centroids, variances, temperature=.05, gamma=15.,
                    chunk_size=4096, gate_cos_min=.2, gate_cos_span=.5,
                    max_night_weight=.5):
    """Average night and day-replay Eq. (8) losses with detached diagnostics.

    Both ID vectors index the original global GSV bank. The night ID is the
    exact source day's place ID; it is never renumbered within a night subset.
    The original daytime covariance remains fixed on both branches.
    Students are unit descriptors from EfficientViTBoQ. Teachers and bank
    arrays are precomputed/frozen. A caller can make one student forward over
    a concatenated night/day batch, then split its output for this objective.
    """
    if (q_night.ndim != 2 or q_replay.ndim != 2 or not len(q_night) or not len(q_replay)
            or q_night.shape[1] != q_replay.shape[1]
            or q_night.shape != teacher_night.shape or q_night.shape != teacher_source_day.shape
            or q_replay.shape != teacher_replay_day.shape
            or night_place_ids.shape != (len(q_night),)
            or replay_place_ids.shape != (len(q_replay),)):
        raise ValueError('Invalid student/teacher/label shapes for night and replay branches')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError('Temperature must be positive and gamma nonnegative')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('Bank chunk size must be a positive integer')
    target, weight, agreement = mix_night_target(teacher_night, teacher_source_day,
        gate_cos_min=gate_cos_min, gate_cos_span=gate_cos_span, max_night_weight=max_night_weight)
    # One global-bank pass for the real mixed batch; group means retain equal
    # night/replay weighting even when the two branch lengths differ.
    losses = asym_loss(torch.cat((q_night, q_replay)),
        torch.cat((target, teacher_replay_day.detach())),
        torch.cat((night_place_ids, replay_place_ids)), centroids, variances,
        temperature, gamma, chunk_size, reduction='none')
    night_loss = losses[:len(q_night)].mean()
    replay_loss = losses[len(q_night):].mean()
    loss = .5 * (night_loss + replay_loss)
    with torch.no_grad():
        metrics = dict(
            night_asym=night_loss.detach(), replay_asym=replay_loss.detach(),
            teacher_daynight_cos=agreement.mean(), night_teacher_weight=weight.mean(),
            qnight_teacherday_cos=F.cosine_similarity(q_night.detach(), teacher_source_day.detach()).mean(),
            qnight_teachernight_cos=F.cosine_similarity(q_night.detach(), teacher_night.detach()).mean(),
        )
    return loss, metrics
