from reproduction.boq_c_distillation_ablation import make_plan
from reproduction.boq_c_distillation import settings
from reproduction.gsv_night_camera import OP_NAMES
from reproduction.boq_c_ablation_resume import ResumeSampler,select_checkpoint
import torch


def test_ablation_changes_only_one_operation_and_adds_resume_adapter(tmp_path):
    plan=make_plan(tmp_path)
    assert len(plan['jobs'])==11
    base,_=settings('configs/boq_dinob_c_distillation.json','control')
    for job in plan['jobs']:
        assert job['arm']=='control'
        config,_=settings(job['config'],job['arm'])
        assert config['gsv_camera_omit']==[job['operation']]
        assert config['distill_weight']==0
        for key in base:
            if key not in ('root','output_dir'):assert config[key]==base[key]
    assert {j['operation'] for j in plan['jobs']}==set(OP_NAMES)
    assert make_plan(tmp_path)==plan


def test_resume_sampler_skips_consumed_batches_and_resets_next_epoch():
    class Sampler:
        epoch=0
        def __len__(self):return 5
        def __iter__(self):return iter([(self.epoch,i) for i in range(5)])
    sampler=ResumeSampler(Sampler(),7)
    assert sampler.epoch==1 and sampler.skip==2
    assert list(sampler)==[(1,2),(1,3),(1,4)]
    sampler.epoch=2
    assert list(sampler)==[(2,i) for i in range(5)]
    assert len(sampler)==5


def test_resume_selects_latest_progress_not_filename(tmp_path):
    a,b=tmp_path/'epoch_last.ckpt',tmp_path/'periodic_last.ckpt'
    torch.save({'global_step':900,'optimizer_states':[{'lr':.001}]},a)
    torch.save({'global_step':1000,'optimizer_states':[{'lr':.002}]},b)
    assert select_checkpoint([a,b])==b
    torch.save({'global_step':1100},a)
    assert select_checkpoint([a,b])==a
    assert select_checkpoint([tmp_path/'missing.ckpt']) is None
