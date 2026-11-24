"""No-distillation adaptation selected only by independent SVOX night development R@1."""
import argparse
import json
from pathlib import Path
from lightning.pytorch.callbacks import ModelCheckpoint
from reproduction import boq_c_distillation as training
from reproduction import boq_c_ablation_resume as resume
from reproduction.boq_protocol import BoQTrainingModule
from reproduction.evaluate import atomic_json,sha256
from reproduction.run import atomic_torch_save

MONITOR='svox-night-dev/R@1'


class NoDistillationModule(training.DistillationModule):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.config['distill_weight']!=0:raise ValueError('Only no-distillation trials are permitted')
        # No frozen teacher is moved to GPU or used in any training forward.
        del self.teacher

    def train(self,mode=True):
        return BoQTrainingModule.train(self,mode)

    def training_step(self,batch,batch_idx):
        images,labels,indices,clean,changed=batch
        return BoQTrainingModule.training_step(self,(images,labels,len(indices)),batch_idx)

    def on_train_batch_end(self,outputs,batch,batch_idx):
        return BoQTrainingModule.on_train_batch_end(self,outputs,batch,batch_idx)


class NightTrainer(resume.ResumeTrainer):
    def __init__(self,*args,**kwargs):
        old=kwargs['callbacks'][0]
        # The caller retains this exact object to export best_model_path after fit.
        old.monitor=MONITOR
        old.filename='night-epoch{epoch:02d}'
        super().__init__(*args,**kwargs)


def choose_initial(initial_score,trained_score):
    return initial_score>=trained_score


def recover_finished_export(out,config):
    """Recover a failed export from a fully trained checkpoint without fitting again."""
    import torch
    last=out/'checkpoints/last.ckpt'
    if (out/'completed.json').exists():return True
    if not last.exists():return False
    state=resume.read_state(last)
    expected=json.loads((out/'experiment.json').read_text())
    if state['identity']!=expected:raise RuntimeError('Recovery identity changed')
    # This experiment's sampler budget is recorded by the immutable queue plan.
    budget=json.loads((out.parents[1]/'plan.json').read_text())['steps_per_trial']
    if state['global_step']<budget:return False
    if state['global_step']!=budget:raise RuntimeError('Recovery budget mismatch')
    saved=[v for v in state['callbacks'].values() if isinstance(v,dict) and v.get('monitor')==MONITOR]
    if len(saved)!=1:raise RuntimeError('Missing night checkpoint callback state')
    path=Path(saved[0]['best_model_path'])
    best=resume.read_state(path)
    if best['identity']!=expected:raise RuntimeError('Best checkpoint identity changed')
    epoch=best['epoch']+1
    score=json.loads((out/f'epoch_{epoch:02d}.json').read_text())['svox-night-dev']['R@1']
    if abs(float(saved[0]['best_model_score'])*100-score)>1e-4:
        raise RuntimeError('Best checkpoint score mismatch')
    weights={k.removeprefix('model.'):v for k,v in best['state_dict'].items() if k.startswith('model.')}
    atomic_torch_save(out/'best.pt',dict(model=weights,identity=expected,epoch=epoch))
    atomic_json(out/'completed.json',dict(steps=budget,identity=expected,best=str(path),
                export_recovery='Recovered retained night-best checkpoint; no training updates'))
    print(f'Recovered completed training export: epoch={epoch}, step={budget}',flush=True)
    return True


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--arm',choices=['control'],default='control')
    args=parser.parse_args()
    config,_=training.settings(args.config,'control')
    if config.get('selection_monitor')!=MONITOR:raise RuntimeError('Missing night selection policy')
    out=Path(config['output_dir'])
    if (out/'night_selection.json').exists() and (out/'completed.json').exists():
        receipt=json.loads((out/'night_selection.json').read_text())
        if sha256(out/'best.pt')!=receipt['weights_sha256']:raise RuntimeError('Selected weights changed')
        return
    training.Trainer=NightTrainer;training.DistillationModule=NoDistillationModule
    training.resume_epoch=resume.resume_epoch
    if not recover_finished_export(out,config):training.main()
    out=Path(config['output_dir'])
    import torch
    selected=torch.load(out/'best.pt',map_location='cpu',weights_only=False,mmap=True)
    epoch=selected['epoch']
    metrics=json.loads((out/f'epoch_{epoch:02d}.json').read_text())
    initial=json.loads((out/'baseline.json').read_text())['svox-night-dev']['R@1']
    score=metrics['svox-night-dev']['R@1']
    use_initial=choose_initial(initial,score)
    selected_path=config['checkpoint'] if use_initial else str(out/'best.pt')
    if use_initial:
        original=torch.load(config['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
        atomic_torch_save(out/'best.pt',dict(model=original['model'],identity=selected['identity'],
                          epoch=0,step=0,selection=MONITOR,score=initial))
    receipt=dict(monitor=MONITOR,score_percent=initial if use_initial else score,
        epoch=0 if use_initial else epoch,initial_score_percent=initial,trained_best_score_percent=score,
        trained_best_epoch=epoch,selected_initial=use_initial,best_weights=str(out/'best.pt'),
        weights_sha256=sha256(out/'best.pt'),runner_sha256=sha256(__file__),
        policy='Best SVOX night dev R@1; earliest tie; initial model is eligible; tests never select')
    atomic_json(out/'night_selection.json',receipt)
    done=json.loads((out/'completed.json').read_text())
    done.update(best_training_checkpoint=done['best'],best=str(out/'best.pt'),selection=receipt)
    atomic_json(out/'completed.json',done)
    print('NIGHT SELECTION '+json.dumps(receipt),flush=True)


if __name__=='__main__':main()
