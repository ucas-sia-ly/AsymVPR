import torch
from reproduction.boq_c_distillation import distillation_loss, collate, settings


def test_distillation_only_changes_selected_student_and_detaches_teacher():
    pred=torch.randn(4,7,requires_grad=True)
    target=torch.randn(4,7,requires_grad=True)
    mask=torch.tensor([True,False,True,False])
    loss=distillation_loss(pred,target,mask)
    loss.backward()
    assert target.grad is None
    assert pred.grad[mask].abs().sum()>0
    assert pred.grad[~mask].abs().sum()==0
    assert torch.isfinite(loss)


def test_no_degradation_is_differentiable_zero():
    pred=torch.randn(4,7,requires_grad=True)
    loss=distillation_loss(pred,pred.detach(),torch.zeros(4,dtype=torch.bool))
    loss.backward()
    assert loss.item()==0 and pred.grad.abs().sum()==0


def test_collation_aligns_clean_teacher_images_with_night_indices():
    a=torch.zeros(3,4,4);b=torch.ones_like(a)
    images,labels,idx,clean,changed=collate([(a,2,False,None,False),(b,2,True,a,True)],2)
    assert images.shape==(1,2,3,4,4)
    assert labels.tolist()==[[2,2]] and idx.tolist()==[1]
    assert torch.equal(clean[0],a) and changed.tolist()==[True]


def test_control_and_distillation_share_recipe_except_weight_and_output():
    a,_=settings('configs/boq_dinob_c_distillation.json','control')
    b,_=settings('configs/boq_dinob_c_distillation.json','distill')
    assert a['distill_weight']==0 and b['distill_weight']==1
    for key in a:
        if key not in ('arm','output_dir','distill_weight'):assert a[key]==b[key]
    assert b['descriptor_dim']==8192 and b['gsv_camera']=='adjusted'
