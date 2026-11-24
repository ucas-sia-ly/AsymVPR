# Copyright (c) 2024 Amar Ali-bey. MIT, see LICENSE.boq.
# Optimizer, warmup, loss and transforms adapted from Bag-of-Queries commit
# 1a4965ea7dfd9bd0dd846adf7a0e430f68101d12 (train.py, src/model.py,
# src/dataloaders/datamodule.py). Local changes provide logging and data adapters.
"""The official BoQ training recipe, shared by real and generated-view trials."""
import time
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torchvision
from pytorch_metric_learning import losses, miners
from torch.utils.data import Dataset
from torchvision.transforms import v2 as T

from reproduction.evaluate import atomic_json, recall_at_k
from src.backbones import DinoV2
from src.boq import BoQ


UPSTREAM_COMMIT = '1a4965ea7dfd9bd0dd846adf7a0e430f68101d12'
OFFICIAL = dict(recipe='boq_official_v1', seed=2024, batch_size=128, img_per_place=4,
                epochs=40, warmup_epochs=10, lr=1e-4, weight_decay=1e-4,
                milestones=[10, 20], lr_mul=.1, precision='16-mixed',
                train_image_size=224, image_size=322, proj_channels=512,
                descriptor_dim=8192, num_queries=64, num_layers=2,
                unfreeze_n_blocks=2)


def check_recipe(config):
    """No unproven optimizer/batch/augmentation changes in the official baseline."""
    for name, expected in OFFICIAL.items():
        if config.get(name) != expected:
            raise ValueError(f'Official BoQ requires {name}={expected!r}; '
                             'legacy trials must keep their historical records')


def input_transform(size, train=False):
    # Upstream decodes RGB uint8 tensors, resizes before casting, and uses
    # RandAugment only for training. This differs from the asymmetric teacher path.
    operations = [T.Resize((size, size), interpolation=3)]
    if train:
        operations.append(T.RandAugment(num_ops=3, magnitude=15, interpolation=2))
    return T.Compose(operations + [T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])])


class ValidationImages(Dataset):
    def __init__(self, benchmark, size):
        self.benchmark = benchmark
        self.paths = benchmark.references + benchmark.queries
        self.transform = input_transform(size)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = torchvision.io.decode_image(str(self.paths[index]), mode='RGB')
        return self.transform(image), index


class MixedBoQ(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        dino = torch.hub.load(config['dino_source'], 'dinov2_vitb14',
                              source='local', pretrained=False)
        dino.load_state_dict(torch.load(config['backbone_weights'],
                                       map_location='cpu', weights_only=True), strict=True)
        self.backbone = DinoV2(dino=dino, unfreeze_n_blocks=config['unfreeze_n_blocks'])
        self.aggregator = BoQ(in_channels=dino.embed_dim,
            proj_channels=config['proj_channels'], num_queries=config['num_queries'],
            num_layers=config['num_layers'], row_dim=config['descriptor_dim']//config['proj_channels'])

    def forward(self, images):
        # Preserve the upstream attention path in training AND validation.
        return self.aggregator(self.backbone(images))[0]


class BoQTrainingModule(L.LightningModule):
    def __init__(self, model, config, identity, benchmarks, logger):
        super().__init__()
        self.model, self.config, self.identity = model, config, identity
        self.benchmarks, self.run_logger = benchmarks, logger
        self.ms_loss = losses.MultiSimilarityLoss(alpha=1, beta=50, base=0.)
        self.ms_miner = miners.MultiSimilarityMiner(epsilon=.1)
        self.validation_outputs = {}
        self.window = None

    def configure_optimizers(self):
        # Same parameter groups and equal learning rates as upstream src/model.py.
        optimizer_params = [
            {'params': self.model.backbone.parameters(), 'lr': self.config['lr'],
             'weight_decay': self.config['weight_decay']},
            {'params': self.model.aggregator.parameters(), 'lr': self.config['lr'],
             'weight_decay': self.config['weight_decay']},
        ]
        optimizer = torch.optim.AdamW(optimizer_params)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=self.config['milestones'], gamma=self.config['lr_mul'])
        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.current_epoch < self.config['warmup_epochs']:
            total_warmup_steps = self.config['warmup_epochs'] * self.trainer.num_training_batches
            lr_scale = min(1., (self.trainer.global_step + 1) / total_warmup_steps)
            for group in optimizer.param_groups:
                initial_lr = group.get('initial_lr', self.config['lr'])
                group['lr'] = lr_scale * initial_lr
        optimizer.step(closure=optimizer_closure)

    def on_train_epoch_start(self):
        self.window = dict(loss=None, batches=0, real=0, generated=0, pairs=0,
                           start=time.monotonic())

    def on_train_epoch_end(self):
        # Lightning creates/prefetches the next DataLoader iterator before the
        # next epoch-start hook. Refresh here, like upstream _refresh_dataframes.
        self.trainer.train_dataloader.batch_sampler.epoch = self.current_epoch + 1

    def training_step(self, batch, batch_idx):
        images, labels, synthetic = batch
        # Upstream P places x K views, with all P*K descriptors mined together.
        descriptors = self.model(images.flatten(0, 1))
        labels = labels.flatten()
        mined_pairs = self.ms_miner(descriptors, labels)
        loss = self.ms_loss(descriptors, labels, mined_pairs)
        self.log('loss', loss, batch_size=len(labels))
        detached = loss.detach()
        self.window['loss'] = detached if self.window['loss'] is None else self.window['loss'] + detached
        self.window['batches'] += 1
        self.window['generated'] += synthetic
        self.window['real'] += len(labels) - synthetic
        self.window['pairs'] += mined_pairs[0].numel()
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = self.global_step
        if step == 1 or step % self.config['log_every'] == 0 or self.trainer.is_last_batch:
            window = self.window
            count = window['real'] + window['generated']
            opt = self.trainer.optimizers[0]
            self.run_logger.log(step, **{
                'train/loss': float(window['loss']/window['batches']),
                'train/epoch': self.current_epoch + 1,
                'train/generated_fraction': window['generated']/count,
                'train/real_images_per_batch': window['real']/window['batches'],
                'train/generated_images_per_batch': window['generated']/window['batches'],
                'train/mined_positive_pairs_per_batch': window['pairs']/window['batches'],
                'train/images_per_second': count/(time.monotonic()-window['start']),
                'lr/backbone': opt.param_groups[0]['lr'], 'lr/boq': opt.param_groups[1]['lr']})
            self.window = dict(loss=None, batches=0, real=0, generated=0, pairs=0,
                               start=time.monotonic())

    def on_validation_epoch_start(self):
        self.validation_outputs = {}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        import faiss
        images, indices = batch
        values = self.model(images).detach().float().cpu().numpy()
        if not np.isfinite(values).all():
            raise ValueError('Invalid validation descriptors')
        benchmark = self.benchmarks[dataloader_idx]
        nrefs = len(benchmark.references)
        state = self.validation_outputs.setdefault(dataloader_idx,
            dict(index=faiss.IndexFlatL2(values.shape[1]), predictions=[], rows=0))
        start = state['rows']
        if not torch.equal(indices.cpu(), torch.arange(start, start+len(values))):
            raise ValueError('Validation image order changed')
        refs = max(0, min(len(values), nrefs-start))
        if refs:
            state['index'].add(values[:refs])
        if refs < len(values):
            if state['index'].ntotal != nrefs:
                raise ValueError('Incomplete validation gallery')
            _, predictions = state['index'].search(values[refs:], 10)
            state['predictions'].append(predictions)
        state['rows'] += len(values)
        if state['rows'] == nrefs + len(benchmark.queries):
            # Release this gallery before building the next dataset's index.
            state['index'] = None

    def on_validation_epoch_end(self):
        metrics = {}
        for index, state in self.validation_outputs.items():
            benchmark = self.benchmarks[index]
            if state['rows'] != len(benchmark.references)+len(benchmark.queries):
                raise ValueError('Incomplete validation descriptors')
            predictions = np.concatenate(state['predictions'])
            values = recall_at_k(predictions, benchmark.positives, (1, 5, 10))
            metrics[benchmark.name] = values
            for key, value in values.items():
                self.log(f'{benchmark.name}/{key}', value/100., add_dataloader_idx=False)
            self.run_logger.log(self.global_step, **{
                f'val/{benchmark.name}/{key}': value for key, value in values.items()})
        self.validation_outputs.clear()
        atomic_json(Path(self.config['output_dir'])/f'epoch_{self.current_epoch+1:02d}.json', metrics)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['identity'] = self.identity

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get('identity') != self.identity:
            raise RuntimeError('Resume experiment identity changed')
