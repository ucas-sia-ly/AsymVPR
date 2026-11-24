"""Wait for authorized ablations and evaluate all remaining fixed final-epoch models."""
import argparse
import fcntl
import json
from pathlib import Path
import time

from reproduction import boq_c_night_evaluation as evaluation
from reproduction import boq_c_darkdriving_recovery as recovery
from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json,sha256
from reproduction.gsv_night_camera import OP_NAMES

OUT=Path('runs/boq_c_remaining_ablations_night_v1')
OLD=Path('runs/boq_c_multi_night_epoch10_v2/combined_summary.json')
REMAINING=[op for op in OP_NAMES if op not in ('saturation','white_balance','defocus')]


def remaining_specs():
    return {f'no_{op}':evaluation.ABL/op/'control/checkpoints/last.ckpt' for op in REMAINING}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for',required=True);args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'.followup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(OUT/'scheduled.json',dict(wait_for=args.wait_for,models=list(remaining_specs()),
            epochs=10,protocol_count=10,pairs=80,selection='fixed epoch 10; do not select on night tests',
            source_sha256=sha256(__file__)))
        print(f'等待消融队列 {args.wait_for} 完成，之后补测8个模型×10项协议。',flush=True)
        while True:
            run=helper('status','--id',args.wait_for)['run']
            if run['status']=='completed' and run.get('exit_code')==0:break
            if run['status'] not in ('running','queued','stopping'):raise RuntimeError('Training queue interrupted; evaluation not started')
            time.sleep(20)
        for op in OP_NAMES:
            path=evaluation.ABL/op/'control/completed.json'
            done=json.loads(path.read_text())
            if done['steps']!=9770:raise RuntimeError('Incomplete training budget')
        evaluation.OUT=OUT;evaluation.SPECS=remaining_specs()
        recovery.OUT=OUT
        evaluation.ValidationImages=recovery.StrictFallbackImages
        evaluation.TRANSFORM += '/strict-Pillow-fallback-on-truncated-JPEG:'+sha256(recovery.__file__)
        evaluation.main()
        old=json.loads(OLD.read_text());new=json.loads((OUT/'summary.json').read_text())
        if old['protocols']!=new['protocols']:raise RuntimeError('Night protocols changed')
        for name,rows in new['results'].items():old['results'].setdefault(name,{}).update(rows)
        old['models'].update(new['models'])
        old.update(status='completed',completed_pairs=sum(len(v) for v in old['results'].values()),
            consolidated_sources={str(p):sha256(p) for p in (OLD,OUT/'summary.json')})
        atomic_json(OUT/'all_models_summary.json',old)
        evaluation.SPECS=old['models'];evaluation.OUT=OUT/'all_models';evaluation.OUT.mkdir(exist_ok=True)
        evaluation.report(old)
        print(f"全部 {old['completed_pairs']} 项模型×协议评测完成。",flush=True)


if __name__=='__main__':main()
