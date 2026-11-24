"""Resume adapter for future no-distillation ablations; leaves active full-C code untouched."""
import itertools
from pathlib import Path
import random
import numpy as np
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from reproduction import boq_c_distillation as training


def read_state(path):
    return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def select_checkpoint(paths):
    existing=[Path(p) for p in paths if Path(p).is_file()]
    return max(existing,key=lambda p:read_state(p)['global_step']) if existing else None


class ResumeSampler:
    def __init__(self,sampler,step):
        self.sampler=sampler
        self.sampler.epoch=step//len(sampler)
        self.skip=step%len(sampler)

    @property
    def epoch(self):return self.sampler.epoch

    @epoch.setter
    def epoch(self,value):
        if value!=self.sampler.epoch:self.skip=0
        self.sampler.epoch=value

    def __len__(self):return len(self.sampler)

    def __iter__(self):return itertools.islice(iter(self.sampler),self.skip,None)


class RandomState(Callback):
    def on_save_checkpoint(self,trainer,pl_module,checkpoint):
        checkpoint['ablation_rng']=dict(python=random.getstate(),numpy=np.random.get_state(),
            torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

    def on_load_checkpoint(self,trainer,pl_module,checkpoint):
        self.pending=checkpoint.get('ablation_rng')

    def on_train_start(self,trainer,pl_module):
        state=getattr(self,'pending',None)
        if state:
            random.setstate(state['python']);np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
            self.pending=None


OriginalTrainer=training.Trainer


class ResumeTrainer(OriginalTrainer):
    def __init__(self,*args,**kwargs):
        callbacks=kwargs.setdefault('callbacks',[])
        self.resume_dir=Path(callbacks[0].dirpath).parent/'resume'
        callbacks.extend([RandomState(),ModelCheckpoint(dirpath=self.resume_dir,save_top_k=0,
            save_last=True,every_n_train_steps=100,save_on_train_epoch_end=False)])
        super().__init__(*args,**kwargs)

    def fit(self,model,*args,**kwargs):
        out=Path(model.config['output_dir'])
        path=select_checkpoint([out/'checkpoints/last.ckpt',out/'resume/last.ckpt'])
        if path:
            state=read_state(path)
            if state.get('identity')!=model.identity:raise RuntimeError('Resume identity changed')
            loader=kwargs['train_dataloaders']
            sampler=ResumeSampler(loader.batch_sampler,state['global_step'])
            object.__setattr__(loader,'batch_sampler',sampler)
            kwargs['ckpt_path']=str(path)
            print(f'Resuming full state: {path}, step={state["global_step"]}, skip={sampler.skip}',flush=True)
        return super().fit(model,*args,**kwargs)


def resume_epoch(path,identity,steps):
    if not Path(path).is_file():return 0
    state=read_state(path)
    if state.get('identity')!=identity:raise RuntimeError('Resume identity changed')
    return state['global_step']//steps


def main():
    training.Trainer=ResumeTrainer
    training.resume_epoch=resume_epoch
    training.main()


if __name__=='__main__':main()
