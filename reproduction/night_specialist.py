"""Night-query-only data, objective and selection for a separate experiment.

The student serves night queries against the original frozen teacher gallery.
Day queries never enter its forward pass. Source-day descriptors can be logged
as diagnostics but do not form a positive target or a training constraint.
"""
import math

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from reproduction.asym_model import asym_loss
from reproduction.data import transform


def validate_night_only_config(config):
    expected = dict(query_domain='night_only', precision='float32',
                    night_target='teacher_night',
                    selection_policy='svox_night_r1')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'Night-only experiment requires {key}={value!r}')
    batch = config.get('night_batch_size')
    if type(batch) is not int or batch <= 0:
        raise ValueError('night_batch_size must be a positive integer')
    if type(config.get('batch_size')) is not int or config['batch_size'] != batch:
        raise ValueError('Night-only batch_size must equal night_batch_size')
    if type(config.get('replay_batch_size')) is not int or config['replay_batch_size'] != 0:
        raise ValueError('Night-only experiment requires replay_batch_size=0')
    if 'max_msls_drop_pp' not in config or config['max_msls_drop_pp'] is not None:
        raise ValueError('Night-only experiment requires max_msls_drop_pp=None; MSLS is diagnostic')
    if type(config.get('epochs')) is not int or config['epochs'] <= 0:
        raise ValueError('epochs must be a positive integer')


class NightOnlyQueries(Dataset):
    """One deterministic night pass, retaining original teacher rows/place IDs.

    No day-image path, day-replay schedule or base-cache image list is opened.
    The order matches PairedReplay's night order for the same seed and epoch.
    Metadata is supplied by the same immutable, place-split night snapshot.
    """
    def __init__(self, records, all_records, config, epoch):
        self.records = records
        self.all_index = {}
        for index, row in enumerate(all_records):
            name = row['night_path']
            if name in self.all_index:
                raise ValueError('Duplicate night path in the all-record teacher order')
            self.all_index[name] = index
        for row in records:
            index = self.all_index.get(row['night_path'])
            if index is None or all_records[index] != row:
                raise ValueError('Night query differs from its immutable teacher snapshot row')
            if (type(row['label']) is not int or row['label'] < 0
                    or type(row['source_index']) is not int or row['source_index'] < 0):
                raise ValueError('Night queries require nonnegative global labels/source indices')
            if row.get('split') == 'val':
                raise ValueError('Held-out night queries cannot enter the training dataset')
        rng = np.random.default_rng(config['seed'] + epoch)
        self.order = rng.permutation(len(records))
        self.transform = transform(config['image_size'])

    def __len__(self):
        return len(self.order)

    def __getitem__(self, index):
        row = self.records[int(self.order[index])]
        with Image.open(row['night_path']) as image:
            night = self.transform(image.convert('RGB'))
        return night, self.all_index[row['night_path']], row['source_index'], row['label']


def night_only_objective(qnight, *, teacher_night, teacher_source_day=None,
                         night_place_ids, centroids, variances,
                         temperature=.05, gamma=15., chunk_size=4096):
    """Pure detached T(night) positive and fixed day-gallery Eq. (8) negatives.

    Unit-normalized teacher descriptors come from the verified frozen cache.
    There is no replay loss, 0.5 scaling, positive blending or agreement gate.
    The optional source-day tensor affects detached diagnostic scalars only.
    """
    if (qnight.ndim != 2 or not len(qnight) or qnight.shape != teacher_night.shape
            or night_place_ids.shape != (len(qnight),)
            or (teacher_source_day is not None and teacher_source_day.shape != qnight.shape)):
        raise ValueError('Invalid night student/teacher/global-label shapes')
    if not math.isfinite(temperature) or temperature <= 0 or not math.isfinite(gamma) or gamma < 0:
        raise ValueError('Temperature must be positive and gamma nonnegative')
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('Bank chunk size must be a positive integer')
    target = teacher_night.detach()
    loss = asym_loss(qnight, target, night_place_ids, centroids.detach(), variances.detach(),
                     temperature, gamma, chunk_size)
    with torch.no_grad():
        metrics = dict(night_asym=loss.detach(),
            qnight_teachernight_cos=F.cosine_similarity(qnight.detach(), target).mean())
        if teacher_source_day is not None:
            day = teacher_source_day.detach()
            metrics.update(
                teacher_daynight_cos=F.cosine_similarity(day, target).mean(),
                qnight_teacherday_cos=F.cosine_similarity(qnight.detach(), day).mean())
    return loss, metrics


def select_night_best(metrics, baseline):
    """Return eligibility/real-night score/day diagnostic, with no day guard.

    The caller replaces best only when the score strictly improves its current
    best, retaining the initial checkpoint if no adaptation improves night dev.
    """
    score = float(metrics['svox_dev']['metrics_percent']['R@1'])
    day = float(metrics['msls']['metrics_percent']['R@1'])
    baseline_day = float(baseline['msls']['metrics_percent']['R@1'])
    if any(not math.isfinite(value) or not 0 <= value <= 100 for value in (score, day, baseline_day)):
        raise ValueError('Night selection and diagnostic Recall values must be finite percentages')
    return True, score, float(day - baseline_day)
