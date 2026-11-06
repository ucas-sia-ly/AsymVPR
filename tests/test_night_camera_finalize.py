"""Tiny CPU records cover the automatic camera-off/on report and reuse."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from reproduction.benchmarks import DATASETS, json_sha256
from reproduction.evaluate import load_inference_checkpoint, sha256
from reproduction.night_finalize import _sealed, compare_summaries
from reproduction.night_camera_finalize import finalize_camera, validate_night_reference
from test_night_camera import with_camera
import test_night_finalize as finalization_fixture


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def seal_run(config, initial, score):
    train=Path(config['output_dir']);root=train.parent
    state=load_inference_checkpoint(train/'best.pt')
    state['config']=config
    state['metrics']['svox_dev']['metrics_percent']['R@1']=score
    torch.save(state,train/'best.pt')
    selected=dict(epoch=state['epoch'],step=state['step'],score=score,adapted=True)
    completed=dict(status='completed',epochs=10,steps=9080,config=config,best=selected)
    write(train/'completed.json',completed)
    summary=deepcopy(initial)
    summary.update(checkpoint=str((train/'best.pt').resolve()),checkpoint_sha256=sha256(train/'best.pt'),
        precision='float32',tf32=False)
    outputs={}
    for name in DATASETS:
        summary['datasets'][name]['metrics_percent']['R@1']=score
        pred=root/'final_evaluation'/f'{name}_top10.npy';pred.parent.mkdir(parents=True,exist_ok=True)
        np.save(pred,np.tile(np.arange(10),(2,1)))
        summary['datasets'][name]['predictions_sha256']=sha256(pred)
        outputs[str(pred.resolve())]=sha256(pred)
    summary_path=root/'final_evaluation/summary.json';write(summary_path,summary)
    comparison=dict(training_config=config,selected=selected,adapted=True,
        svox_dev=dict(selected_percent=state['metrics']['svox_dev']['metrics_percent']),
        five_benchmark_comparison=compare_summaries(initial,summary),tokyo_night={'available':False})
    cmp=root/'final_evaluation/comparison.json';write(cmp,_sealed(comparison))
    for p in (summary_path,cmp):outputs[str(p.resolve())]=sha256(p)
    identity=dict(best_sha256=sha256(train/'best.pt'),config_sha256=json_sha256(config),
        training_completed_sha256=sha256(train/'completed.json'))
    receipt=dict(status='complete',comparison=comparison,identity=identity,identity_sha256=json_sha256(identity),
        outputs_sha256=outputs)
    write(root/'finalization.json',_sealed(receipt))
    return summary_path,cmp


@pytest.fixture
def camera_records(tmp_path):
    config,source,cache,benches,initial=finalization_fixture.FinalizationTests().fixture(tmp_path)
    actual=json.loads(Path('configs/boq_b2_night_only_v1.json').read_text())
    old={**actual,**config,'query_domain':'night_only','selection_policy':'svox_night_r1',
        'night_target':'teacher_night','replay_batch_size':0,'max_msls_drop_pp':None}
    summary,comparison=seal_run(old,initial,56.)
    train=tmp_path/'camera/train';train.mkdir(parents=True)
    state=load_inference_checkpoint(Path(old['output_dir'])/'best.pt')
    state.update(epoch=3,step=9)
    torch.save(state,train/'best.pt')
    config=with_camera(dict(old,output_dir=str(train),night_only_reference_summary=str(summary),
        night_only_reference_comparison=str(comparison),
        night_only_reference_summary_sha256=sha256(summary),
        night_only_reference_comparison_sha256=sha256(comparison)),tmp_path)
    seal_run(config,initial,57.)
    return config,benches


def test_completed_camera_off_on_report_seals_and_reuses_without_new_inference(camera_records):
    config,benches=camera_records
    reference=validate_night_reference(config)
    assert reference['selected']['score']==56.
    with patch('reproduction.night_camera_finalize._benchmarks',return_value=benches), \
         patch('reproduction.night_camera_finalize.tokyo_night_diagnostic',return_value={'available':False}), \
         patch('subprocess.run') as inference:
        first=finalize_camera(config)
        again=finalize_camera(config)
    inference.assert_not_called()
    assert first['identity']==again['identity']
    assert first['comparison']['svox_dev']['difference_percentage_points']['R@1']==1.
    assert first['comparison']['five_benchmark_comparison']['msls']['difference_percentage_points']['R@1']==1.
    report=Path(config['output_dir']).parent/'CAMERA_REPORT.md'
    assert 'ToTensor → Resize' in report.read_text()
    assert '未退化夜图' in report.read_text()
    report.write_text('corrupt')
    with pytest.raises(ValueError,match='sealed outputs changed'):
        finalize_camera(config)


def test_camera_reference_rejects_changed_budget_hash_and_original_output(camera_records):
    config,_=camera_records
    for key,value,match in [('night_only_reference_summary_sha256','bad','SHA256'),
            ('epochs',11,'controlled field'),('lr_backbone',.1,'controlled field')]:
        with pytest.raises(ValueError,match=match):
            validate_night_reference(dict(config,**{key:value}))
    oldroot=Path(config['night_only_reference_summary']).parent.parent
    with pytest.raises(ValueError,match='outside'):
        validate_night_reference(dict(config,output_dir=str(oldroot/'train')))
