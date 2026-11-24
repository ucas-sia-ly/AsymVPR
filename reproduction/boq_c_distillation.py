"""Paired C-policy BoQ finetuning: MS only versus MS + clean-teacher distillation."""
import argparse
from functools import partial
import fcntl
import json
import os
from pathlib import Path
import time

import lightning as L
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from reproduction.boq_protocol import BoQTrainingModule, MixedBoQ, ValidationImages
from reproduction.data import transform
from reproduction.evaluate import atomic_json, sha256
from reproduction.gsv_night_camera import GSVCamera
from reproduction.monitoring import TrainingLogger
from reproduction.night_data import load_records
from reproduction.night_eval import load_svox_dev
from reproduction.benchmarks import Benchmark
from reproduction.train_dino_gsv_mixed import MixedViews, EpochBatches, load_config, validation_sets, resume_epoch
from reproduction.run import configure_cuda, atomic_torch_save

DEFAULT = 'configs/boq_dinob_c_distillation.json'


class PairedViews(MixedViews):
    def __init__(self, config):
        rows = load_records(config, 'train')
        super().__init__(config, records=rows)
        self.rows = {str(Path(row['night_path']).resolve()): row for row in rows}
        self.config = config
        self.clean = transform(config['train_image_size'])
        self.camera = None

    def __getitem__(self, key):
        index, epoch = key
        if index >= 0:
            image, label, _ = super().__getitem__(index)
            return image, label, False, None, False
        path = self.night_paths[-1-index]
        if self.camera is None or self.camera.epoch != epoch:
            self.camera = GSVCamera(self.config, epoch, self.clean)
        with Image.open(path) as image:
            image = image.convert('RGB')
            clean = self.clean(image)
            degraded, flags = self.camera.apply(image, self.rows[path])
        return degraded, int(self.generated_labels[-1-index]), True, clean, bool(flags[0])


class PairedBatches(EpochBatches):
    def __iter__(self):
        epoch = self.epoch
        for batch in self.data.batches(self.seed, epoch, self.ratio):
            yield [(i, epoch) for i in batch]


def collate(batch, k):
    images = torch.stack([row[0] for row in batch]).reshape(-1,k,*batch[0][0].shape)
    labels = torch.tensor([row[1] for row in batch]).reshape(-1,k)
    night_indices = torch.tensor([i for i,row in enumerate(batch) if row[2]], dtype=torch.long)
    clean = torch.stack([row[3] for row in batch if row[2]]) if len(night_indices) else images.new_empty((0,*images.shape[2:]))
    changed = torch.tensor([row[4] for row in batch if row[2]], dtype=torch.bool)
    return images, labels, night_indices, clean, changed


def distillation_loss(prediction, target, changed):
    if not changed.any():
        return prediction.sum()*0
    return (1-F.cosine_similarity(prediction[changed].float(), target[changed].detach().float(),dim=-1)).mean()


class DistillationModule(BoQTrainingModule):
    def __init__(self, model, teacher, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.teacher = teacher.requires_grad_(False).eval()
        self.kd_window = []

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def training_step(self, batch, batch_idx):
        images, labels, indices, clean, changed = batch
        outputs = self.model(images.flatten(0,1))
        labels = labels.flatten()
        mined = self.ms_miner(outputs, labels)
        ms = self.ms_loss(outputs, labels, mined)
        with torch.no_grad():
            targets = self.teacher(clean) if len(clean) else outputs[:0]
        kd = distillation_loss(outputs[indices], targets, changed)
        loss = ms + self.config['distill_weight']*kd
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite distillation loss')
        self.log('loss',loss,batch_size=len(labels))
        w = self.window
        w['loss'] = loss.detach() if w['loss'] is None else w['loss']+loss.detach()
        w['batches'] += 1; w['generated'] += len(indices); w['real'] += len(labels)-len(indices)
        w['pairs'] += mined[0].numel()
        self.kd_window.append(torch.stack((ms.detach().float(),kd.detach().float(),changed.float().sum())))
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.global_step == 1 or self.global_step % self.config['log_every'] == 0 or self.trainer.is_last_batch:
            stats = torch.stack(self.kd_window).mean(0).tolist()
            self.run_logger.log(self.global_step, **{'train/ms_loss':stats[0], 'train/distillation_loss':stats[1],
                'train/weighted_distillation_loss':stats[1]*self.config['distill_weight'],
                'train/degraded_night_images_per_batch':stats[2]})
            self.kd_window.clear()
        super().on_train_batch_end(outputs,batch,batch_idx)


def settings(path, arm):
    plan = json.loads(Path(path).read_text())
    config = load_config(plan['training_config'])
    c = json.loads(Path(plan['c_config']).read_text())
    for key in ('night_release_root','svox_root','svox_dev_manifest','gsv_camera','gsv_edge_threshold'):
        config[key] = c[key]
    config.update(plan['finetuning'])
    config.update(root=plan['root'], checkpoint=plan['checkpoint'], ratio='8:1', arm=arm,
                  distill_weight=1. if arm=='distill' else 0., output_dir=str(Path(plan['root'])/arm))
    return config, plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT)
    parser.add_argument('--arm',choices=['distill','control'],required=True)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    config,plan=settings(args.config,args.arm)
    torch.set_num_threads(config['cpu_threads'])
    data=PairedViews(config)
    if args.preflight:
        sample=next(iter(PairedBatches(data,config['seed'],'8:1')))
        night=[key for key in sample if key[0]<0]
        pair=data[night[0]]
        assert pair[0].shape==pair[3].shape==(3,322,322)
        print(json.dumps(dict(places=len(data.places),steps=data.steps*config['epochs'],
              night_images_per_full_batch=len(night),shape=list(pair[0].shape))))
        return
    out=Path(config['output_dir']);out.mkdir(parents=True,exist_ok=True)
    with (out/'.train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity=dict(config=config,checkpoint_sha256=sha256(config['checkpoint']),
            c_config_sha256=sha256(plan['c_config']),snapshot_sha256=sha256(Path(config['input_dir'])/'summary.json'),
            source_sha256={p:sha256(p) for p in (__file__,'reproduction/boq_protocol.py',
              'reproduction/train_dino_gsv_mixed.py','reproduction/gsv_night_camera.py','src/boq.py','src/backbones.py')})
        receipt=out/'experiment.json'
        if receipt.exists() and json.loads(receipt.read_text())!=identity:raise RuntimeError('Experiment identity changed')
        if (out/'completed.json').exists():return
        atomic_json(receipt,identity)
        configure_cuda();L.seed_everything(config['seed'],workers=True)
        state=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
        if not Path(config['checkpoint']).with_name('completed.json').is_file():raise RuntimeError('Base training incomplete')
        models=[MixedBoQ(config) for _ in range(2)]
        for model in models:model.load_state_dict(state['model'],strict=True)
        del state
        benches=validation_sets(config)
        svox=load_svox_dev(config)
        benches.append(Benchmark('svox-night-dev',Path(config['svox_root']),svox['references'],svox['queries'],
                                 svox['positives'],svox['definition'],{}))
        for b in benches:atomic_json(out/f'{b.name}_protocol.json',b.describe())
        with TrainingLogger(run_dir=os.environ.get('TRAIN_TERMINAL_RUN_DIR',str(out/'logs')),
                            config=config,total_steps=data.steps*config['epochs']) as logger:
            module=DistillationModule(models[0],models[1],config,identity,benches,logger)
            callback=ModelCheckpoint(dirpath=out/'checkpoints',monitor='msls-val/R@1',mode='max',
                       save_top_k=1,save_last=True,filename='epoch{epoch:02d}')
            trainer=Trainer(accelerator='gpu',devices=[0],precision=config['precision'],max_epochs=config['epochs'],
                callbacks=[callback],logger=TensorBoardLogger(str(out),name='tensorboard'),num_sanity_val_steps=0,
                enable_progress_bar=False,enable_model_summary=False,log_every_n_steps=config['log_every'])
            sampler=PairedBatches(data,config['seed'],'8:1')
            last=out/'checkpoints/last.ckpt';sampler.epoch=resume_epoch(last,identity,data.steps)
            loader=DataLoader(data,batch_sampler=sampler,collate_fn=partial(collate,k=data.k),
                     num_workers=config['num_workers'],prefetch_factor=1,pin_memory=False)
            vals=[DataLoader(ValidationImages(b,config['image_size']),batch_size=64,num_workers=0) for b in benches]
            if not last.exists():
                trainer.validate(module,dataloaders=vals,verbose=False)
                (out/'epoch_01.json').replace(out/'baseline.json')
            trainer.fit(module,train_dataloaders=loader,val_dataloaders=vals,ckpt_path=str(last) if last.exists() else None)
            best=torch.load(callback.best_model_path,map_location='cpu',weights_only=False)
            weights={k.removeprefix('model.'):v for k,v in best['state_dict'].items() if k.startswith('model.')}
            atomic_torch_save(out/'best.pt',dict(model=weights,identity=identity,epoch=best['epoch']+1))
            atomic_json(out/'completed.json',dict(steps=trainer.global_step,identity=identity,best=callback.best_model_path))


if __name__=='__main__':main()
