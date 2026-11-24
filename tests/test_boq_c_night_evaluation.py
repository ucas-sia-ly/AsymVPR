import torch
import pytest
from reproduction.boq_c_night_evaluation import load_weights


def test_final_epoch_adapter_excludes_teacher_parameters(tmp_path):
    path=tmp_path/'last.ckpt'
    torch.save(dict(epoch=9,global_step=9770,state_dict={
        'model.backbone.weight':torch.ones(2),'teacher.backbone.weight':torch.zeros(2)}),path)
    weights,meta=load_weights(path)
    assert set(weights)=={'backbone.weight'}
    assert meta['epoch']==10 and meta['step']==9770


def test_incomplete_training_cannot_enter_final_epoch_comparison(tmp_path):
    path=tmp_path/'last.ckpt'
    torch.save(dict(epoch=1,global_step=1954,state_dict={}),path)
    with pytest.raises(ValueError,match='epoch 10'):load_weights(path)


def test_strict_fallback_preserves_rgb_and_records_evidence(tmp_path,monkeypatch):
    import numpy as np
    from PIL import Image,ImageFile
    from types import SimpleNamespace
    from reproduction import boq_c_darkdriving_recovery as recovery
    path=tmp_path/'image.jpg';Image.new('RGB',(17,19),(40,50,60)).save(path)
    monkeypatch.setattr(recovery,'OUT',tmp_path)
    def truncated(*args):raise RuntimeError('Image is incomplete or truncated')
    monkeypatch.setattr(recovery.OriginalImages,'__getitem__',truncated)
    dataset=recovery.StrictFallbackImages(SimpleNamespace(references=[path],queries=[]),322)
    actual,index=dataset[0]
    with Image.open(path) as im:pixels=np.array(im.convert('RGB'),copy=True)
    expected=dataset.transform(torch.from_numpy(pixels).permute(2,0,1))
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
    assert index==0 and not ImageFile.LOAD_TRUNCATED_IMAGES
    assert (tmp_path/'decoder_fallback.jsonl').is_file()


def test_post_training_plan_covers_only_eight_unmeasured_ablations():
    from reproduction.boq_c_ablation_final_nights import remaining_specs
    from reproduction.gsv_night_camera import OP_NAMES
    specs=remaining_specs()
    assert set(specs)=={f'no_{op}' for op in OP_NAMES if op not in ('saturation','white_balance','defocus')}
    assert len(specs)*10==80
    assert all(p.name=='last.ckpt' for p in specs.values())
