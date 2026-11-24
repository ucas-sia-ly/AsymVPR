"""Test night-dev-selected checkpoints; night test sets never determine selection."""
import argparse
import json
from pathlib import Path
import time
import torch
from reproduction import boq_c_night_evaluation as evaluation
from reproduction import boq_c_darkdriving_recovery as recovery
from reproduction.boq_night_selection_queue import ROOT
from reproduction.gsv_night_camera import OP_NAMES
from reproduction.augmentation_sweep import helper
from reproduction.evaluate import sha256

OUT=Path('runs/boq_night_selected_multi_night_v1')


def specs():
    return dict(baseline=Path('runs/boq_dinob_official_real_memory_v2/real/best.pt'),
        C_no_distill=ROOT/'full_C/control/best.pt',
        **{f'no_{op}':ROOT/op/'control/best.pt' for op in OP_NAMES})


def load_selected(path):
    state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    receipt=path.parent/'night_selection.json'
    if receipt.exists():
        selected=json.loads(receipt.read_text())
        if selected['monitor']!='svox-night-dev/R@1' or selected['weights_sha256']!=sha256(path):
            raise RuntimeError('Night selection receipt mismatch')
        return state['model'],dict(epoch=selected['epoch'],step=selected['epoch']*977,
                   selection=selected['policy'],night_dev_R1=selected['score_percent'])
    return state['model'],dict(epoch=state['epoch'],step=state['step'],selection='pre-adaptation baseline')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for',required=True);args=parser.parse_args()
    print('等待全部12组无蒸馏夜间选模训练完成，再测试13模型×10协议。',flush=True)
    while True:
        r=helper('status','--id',args.wait_for)['run']
        if r['status']=='completed' and r.get('exit_code')==0:break
        if r['status'] not in ('running','queued','stopping'):raise RuntimeError('Training interrupted')
        time.sleep(20)
    for tag in ('full_C',*OP_NAMES):
        d=json.loads((ROOT/tag/'control/completed.json').read_text())
        if d['steps']!=9770:raise RuntimeError('Incomplete training')
    evaluation.OUT=OUT;evaluation.SPECS=specs();evaluation.load_weights=load_selected
    evaluation.ValidationImages=recovery.StrictFallbackImages;recovery.OUT=OUT
    evaluation.TRANSFORM+='/strict-Pillow-fallback:'+sha256(recovery.__file__)
    # Replace the legacy fixed-epoch caption without changing numerical evaluation.
    original_report=evaluation.report
    def report(summary):
        original_report(summary)
        p=OUT/'REPORT.md'
        p.write_text(p.read_text().replace('微调模型固定第10轮，不用测试结果重选权重。',
            '按SVOX夜间开发集R@1选择权重（含初始模型）；测试集不参与选模。'))
    evaluation.report=report
    evaluation.main()


if __name__=='__main__':main()
