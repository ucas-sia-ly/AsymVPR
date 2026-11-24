from types import SimpleNamespace
import torch
from reproduction.boq_night_selection import NoDistillationModule,choose_initial,MONITOR
from reproduction.boq_protocol import OFFICIAL
from reproduction.boq_night_selected_evaluation import specs


def test_no_distillation_module_has_no_teacher_and_metric_loss_only():
    from reproduction.boq_c_distillation import DistillationModule
    model=torch.nn.Module();model.backbone=torch.nn.Linear(2,2);model.aggregator=torch.nn.Linear(2,2)
    module=NoDistillationModule(model,torch.nn.Linear(2,2),dict(OFFICIAL,distill_weight=0.),{},[],SimpleNamespace())
    assert not hasattr(module,'teacher')
    module.train();assert module.model.training
    assert all(not key.startswith('teacher.') for key in module.state_dict())


def test_initial_is_eligible_and_ties_do_not_replace_it():
    assert choose_initial(95.,94.)
    assert choose_initial(95.,95.)
    assert not choose_initial(95.,96.)
    assert MONITOR=='svox-night-dev/R@1'


def test_only_no_distillation_models_enter_final_tests():
    paths=specs()
    assert len(paths)==13
    assert 'C_distill' not in paths
    assert all(p.name=='best.pt' for p in paths.values())


def test_night_callback_reference_survives_fit_and_export(tmp_path):
    import lightning.pytorch as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from reproduction.boq_night_selection import NightTrainer
    class Tiny(L.LightningModule):
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.ones(()))
        def training_step(self,batch,batch_idx):return self.weight.square()
        def validation_step(self,batch,batch_idx):
            self.log(MONITOR,float(2-self.current_epoch),batch_size=1)
        def configure_optimizers(self):return torch.optim.SGD(self.parameters(),lr=.1)
    callback=ModelCheckpoint(dirpath=tmp_path/'checkpoints',monitor='msls-val/R@1',mode='max',save_last=True)
    trainer=NightTrainer(accelerator='cpu',devices=1,max_epochs=2,callbacks=[callback],
        logger=False,enable_progress_bar=False,enable_model_summary=False,num_sanity_val_steps=0)
    data=torch.utils.data.DataLoader(torch.ones(1))
    # Use the underlying Trainer fit; ResumeTrainer requires the full experiment config.
    L.Trainer.fit(trainer,Tiny(),train_dataloaders=data,val_dataloaders=data)
    assert callback in trainer.callbacks
    assert callback.monitor==MONITOR
    best=torch.load(callback.best_model_path,weights_only=False)
    assert best['epoch']==0
    assert callback.best_model_score==2
