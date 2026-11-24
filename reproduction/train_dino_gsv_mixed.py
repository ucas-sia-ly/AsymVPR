"""Official BoQ recipe with optional same-place generated training views.

Only batch composition changes between the real, 8:1 and 4:1 trials. Historical
90-image/two-view trials are not resumed by this recipe. --preflight is CPU only.
"""
import argparse
from functools import partial
import fcntl
import json
import math
import os
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchvision
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

from reproduction.benchmarks import Benchmark, build_benchmark, checked_positives
from reproduction.boq_protocol import (OFFICIAL, UPSTREAM_COMMIT, BoQTrainingModule,
                                      MixedBoQ, ValidationImages, check_recipe, input_transform)
from reproduction.evaluate import atomic_json, sha256
from reproduction.monitoring import TrainingLogger
from reproduction.night_data import load_records
from reproduction.path_resolver import resolve_paths
from reproduction.run import atomic_torch_save, configure_cuda


DEFAULT_CONFIG = 'configs/boq_dinob_official_ratios.json'
RATIOS = ('real', '8:1', '4:1')


def generated_count(ratio, images, image_offset=0):
    """Integer cumulative quotas; error stays below one image for any prefix.

    A 512-image official batch cannot divide exactly into ninths or fifths.
    Alternate adjacent counts without changing P, K, or optimization steps.
    """
    if ratio not in RATIOS:
        raise ValueError('Use real, 8:1 or 4:1; four distinct views per place are required')
    if ratio == 'real':
        return 0
    real, generated = map(int, ratio.split(':'))
    return ((image_offset + images)*generated//(real+generated)
            - image_offset*generated//(real+generated))


def load_config(path):
    settings = json.loads(Path(path).read_text())
    if settings.get('recipe') != OFFICIAL['recipe']:
        raise ValueError(f'Historical nonofficial configuration; use {DEFAULT_CONFIG}')
    base = json.loads(Path(settings['base_config']).read_text())
    keys = ('gsv_root', 'msls_root', 'base_cache', 'input_dir', 'night_release_root', 'dino_source')
    config = {**{k: base[k] for k in keys}, **OFFICIAL, **settings}
    check_recipe(config)
    if config['ratios'] != list(RATIOS):
        raise ValueError('The comparison must include the real-only control, 8:1 and 4:1')
    return config


class MixedViews(Dataset):
    def __init__(self, config, records=None):
        cache = Path(config['base_cache'])
        self.real_paths = np.load(cache/'paths.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(cache/'offsets.npy', mmap_mode='r', allow_pickle=False)
        self.labels = np.load(cache/'labels.npy', mmap_mode='r', allow_pickle=False)
        self.place_keys = json.loads((cache/'place_keys.json').read_text())
        self.root = Path(config['gsv_root']).resolve()
        if Path(json.loads((cache/'manifest.json').read_text())['root']).resolve() != self.root:
            raise ValueError('GSV image root does not match the indexed real views')
        if (self.offsets.ndim != 1 or self.offsets[0] != 0 or self.offsets[-1] != len(self.real_paths)
                or len(self.labels) != len(self.real_paths)
                or len(self.place_keys) != len(self.offsets)-1
                or np.any(np.diff(self.offsets) <= 0)
                or not np.array_equal(np.repeat(np.arange(len(self.offsets)-1), np.diff(self.offsets)), self.labels)):
            raise ValueError('GSV real views and place labels are inconsistent')
        self.k, self.places_per_batch = config['img_per_place'], config['batch_size']
        # Upstream includes ALL GSV places with >= K real views, even those with
        # no generated image. Real-only never narrows to the synthetic subset.
        self.places = np.flatnonzero(np.diff(self.offsets) >= self.k)
        self.city_places = {}
        for label in self.places:
            self.city_places.setdefault(self.place_keys[label][0], []).append(label)
        rows = load_records(config, 'train') if records is None else records
        self.night, self.night_paths = {}, []
        seen = set()
        for row in rows:
            label = row['label']
            if (not 0 <= label < len(self.place_keys)
                    or self.place_keys[label] != [row['city'], row['place_id']]):
                raise ValueError('Generated view has a conflicting city/place label')
            path = str(Path(row['night_path']).resolve())
            if path in seen:
                raise ValueError('Generated image paths must be unique')
            seen.add(path)
            index = len(self.night_paths)
            self.night_paths.append(path)
            self.night.setdefault(label, []).append(-1-index)
        self.steps = math.ceil(len(self.places)/self.places_per_batch)
        if not self.steps:
            raise ValueError('No GSV place has four real views')
        self.transform = input_transform(config['train_image_size'], train=True)

    def batch_layout(self, seed, epoch, ratio):
        # Upstream reshuffles within cities, uses each place once, and keeps the tail.
        rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
        ordered = []
        for labels in self.city_places.values():
            # Upstream shuffles city dataframe ROWS, then takes place IDs in
            # first-occurrence order; uniform shuffling of place IDs differs
            # when places have unequal numbers of real views.
            rows = np.repeat(labels, np.diff(self.offsets)[labels])
            rows = rng.permutation(rows)
            _, first = np.unique(rows, return_index=True)
            ordered.append(rows[np.sort(first)])
        places = np.concatenate(ordered)
        groups = [places[i:i+self.places_per_batch]
                  for i in range(0, len(places), self.places_per_batch)]
        capacities = [sum(label in self.night for label in labels) for labels in groups]
        counts, offset = [], 0
        for labels, capacity in zip(groups, capacities):
            images = len(labels)*self.k
            counts.append(min(capacity, generated_count(ratio, images, offset)))
            offset += images
        deficit = generated_count(ratio, offset) - sum(counts)
        # A tail or city with fewer generated views must not narrow the place pool.
        # Redistribute its small quota deficit over other batches, never duplicate.
        while deficit:
            available = [i for i, (n, capacity) in enumerate(zip(counts, capacities)) if n < capacity]
            if not available:
                raise ValueError(f'{ratio}: insufficient distinct generated views for the full real place pool')
            for i in rng.permutation(available)[:deficit]:
                counts[i] += 1
                deficit -= 1
        return list(zip(groups, counts))

    def batches(self, seed, epoch, ratio, skip=0):
        for batch, (labels, count) in enumerate(self.batch_layout(seed, epoch, ratio)):
            if batch < skip:
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, epoch, batch, 1]))
            views = np.stack([rng.choice(np.arange(self.offsets[label], self.offsets[label+1]),
                                        self.k, replace=False) for label in labels])
            candidates = [i for i in rng.permutation(len(labels)) if labels[i] in self.night]
            for i in candidates[:count]:
                views[i, -1] = rng.choice(self.night[labels[i]])
            yield views.ravel().tolist()

    def __len__(self):
        return len(self.real_paths) + len(self.night_paths)

    def __getitem__(self, index):
        if index < 0:
            path = self.night_paths[-1-index]
            # The negative index is looked up once when the snapshot is indexed.
            label = self.generated_labels[-1-index]
        else:
            label = int(self.labels[index])
            path = self.root/self.real_paths[index].decode()
        image = torchvision.io.decode_image(str(path), mode='RGB')
        return self.transform(image), label, index < 0

    @property
    def generated_labels(self):
        if not hasattr(self, '_generated_labels'):
            self._generated_labels = np.empty(len(self.night_paths), dtype=np.int64)
            for label, indexes in self.night.items():
                for index in indexes:
                    self._generated_labels[-1-index] = label
        return self._generated_labels


class EpochBatches:
    def __init__(self, data, seed, ratio):
        self.data, self.seed, self.ratio, self.epoch = data, seed, ratio, 0

    def __iter__(self):
        return self.data.batches(self.seed, self.epoch, self.ratio)

    def __len__(self):
        return self.data.steps


def collate_places(batch, k):
    images, labels, flags = default_collate(batch)
    return images.reshape(-1, k, *images.shape[1:]), labels.reshape(-1, k), int(flags.sum())


def validation_sets(config):
    msls = build_benchmark('msls', roots={'msls': config['msls_root']})
    msls.name = 'msls-val'
    root = Path(config['pitts30k_root'])
    files = [root/f'pitts30k_val_{suffix}.npy' for suffix in ('dbImages', 'qImages', 'gt')]
    refs, queries = [resolve_paths('pittsburgh', root, np.load(p)) for p in files[:2]]
    positives = checked_positives(np.load(files[2], allow_pickle=True), len(refs), len(queries))
    pitts = Benchmark('pitts30k-val', root, refs, queries, positives,
                      dict(split='validation', positives='published 25m ground truth'),
                      {p.name: sha256(p) for p in files})
    return [msls, pitts]


def trial_summary(data, config, ratio):
    # Metadata only: check every epoch including capacity and the final tail.
    totals = []
    for epoch in range(config['epochs']):
        layout = data.batch_layout(config['seed'], epoch, ratio)
        images = sum(len(labels)*data.k for labels, _ in layout)
        generated = sum(count for _, count in layout)
        full_counts = [count for labels, count in layout if len(labels) == data.places_per_batch]
        totals.append(dict(images=images, real=images-generated, generated=generated,
                           generated_per_full_batch_min=min(full_counts, default=generated),
                           generated_per_full_batch_max=max(full_counts, default=generated)))
    return dict(ratio=ratio, places=len(data.places), places_per_batch=data.places_per_batch,
                images_per_full_batch=data.places_per_batch*data.k, img_per_place=data.k,
                steps_per_epoch=data.steps, total_steps=data.steps*config['epochs'],
                epoch_counts=totals)


def resume_epoch(path, identity, steps_per_epoch):
    if not Path(path).is_file():
        return 0
    state = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if state.get('identity') != identity:
        raise RuntimeError('Resume experiment identity changed')
    step = state['global_step']
    if (type(step) is not int or step % steps_per_epoch
            or step//steps_per_epoch != state['epoch']+1):
        raise RuntimeError('Resume requires a completed epoch checkpoint')
    return step//steps_per_epoch


def run(config, data, identity, out, logger):
    # Same seeding, 16-mixed Lightning trainer, optimizer and checkpoint criterion as upstream.
    L.seed_everything(config['seed'], workers=True)
    benchmarks = validation_sets(config)
    module = BoQTrainingModule(MixedBoQ(config), config, identity, benchmarks, logger)
    checkpointing = ModelCheckpoint(dirpath=out/'checkpoints', monitor='msls-val/R@1',
        filename='epoch[{epoch:02d}]_R@1[{msls-val/R@1:.4f}]_R@5[{msls-val/R@5:.4f}]',
        auto_insert_metric_name=False, save_weights_only=False, save_top_k=3,
        mode='max', save_last=True)
    trainer = Trainer(accelerator='gpu', devices=[0], precision=config['precision'],
        logger=TensorBoardLogger(save_dir=str(out), name='tensorboard', default_hp_metric=False),
        callbacks=[checkpointing], max_epochs=config['epochs'], check_val_every_n_epoch=1,
        num_sanity_val_steps=0, log_every_n_steps=config['log_every'],
        enable_model_summary=False, enable_progress_bar=False)
    options = dict(num_workers=config['num_workers'], pin_memory=False)
    # A bounded prefetch queue changes host memory use, not the P*K metric-learning batch.
    if config['num_workers']:
        options['prefetch_factor'] = 1
    last = out/'checkpoints/last.ckpt'
    sampler = EpochBatches(data, config['seed'], config['ratio'])
    sampler.epoch = resume_epoch(last, identity, data.steps)
    train_loader = DataLoader(data, batch_sampler=sampler,
                              collate_fn=partial(collate_places, k=data.k), **options)
    val_loaders = [DataLoader(ValidationImages(b, config['image_size']),
                             batch_size=config['batch_size'], num_workers=0, pin_memory=False)
                   for b in benchmarks]
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loaders,
                ckpt_path=str(last) if last.is_file() else None)
    state = torch.load(checkpointing.best_model_path, map_location='cpu', weights_only=False)
    weights = {key.removeprefix('model.'): value for key, value in state['state_dict'].items()}
    atomic_torch_save(out/'best.pt', dict(model=weights, identity=identity,
                      epoch=state['epoch']+1, step=state['global_step'],
                      selection='msls-val/R@1', score=float(checkpointing.best_model_score)))
    atomic_json(out/'completed.json', dict(steps=int(trainer.global_step), identity=identity,
                best_checkpoint=checkpointing.best_model_path, selection='msls-val/R@1'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--ratio', choices=RATIOS, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    out = Path(config['root'])/args.ratio.replace(':', 'to')
    config.update(ratio=args.ratio, output_dir=str(out))
    torch.set_num_threads(config['cpu_threads'])
    data = MixedViews(config, records=[] if args.ratio == 'real' else None)
    summary = trial_summary(data, config, args.ratio)
    if args.preflight:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = Path(config['base_cache'])
        identity = dict(config=config, steps_per_epoch=data.steps, places=len(data.places),
            batch_plan=summary, upstream_commit=UPSTREAM_COMMIT,
            backbone_sha256=sha256(config['backbone_weights']),
            snapshot_sha256=(None if args.ratio == 'real' else
                             sha256(Path(config['input_dir'])/'summary.json')),
            real_manifest_sha256={name: sha256(cache/name) for name in
                                 ('manifest.json', 'paths.npy', 'labels.npy', 'offsets.npy', 'place_keys.json')},
            source_sha256={name: sha256(name) for name in (__file__,
                'reproduction/boq_protocol.py', 'src/backbones.py', 'src/boq.py')})
        receipt = out/'experiment.json'
        if receipt.exists() and json.loads(receipt.read_text()) != identity:
            raise RuntimeError('Training lineage changed; use a new output directory')
        if (out/'completed.json').is_file():
            print('Experiment already completed', flush=True)
            return
        atomic_json(receipt, identity)
        configure_cuda()
        with TrainingLogger(run_dir=os.environ.get('TRAIN_TERMINAL_RUN_DIR', str(out/'logs')),
                            config=config, total_steps=summary['total_steps']) as logger:
            run(config, data, identity, out, logger)


if __name__ == '__main__':
    main()
