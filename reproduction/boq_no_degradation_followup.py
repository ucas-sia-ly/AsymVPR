"""Append a matched no-degradation control after the existing ablation queue."""
import argparse
import json
from pathlib import Path
import sys
import fcntl
from reproduction.augmentation_sweep import helper
from reproduction.boq_c_distillation_ablation import wait_for_run,seal
from reproduction.evaluate import atomic_json,sha256
from reproduction.gsv_night_camera import OP_NAMES

ROOT=Path('runs/boq_no_degradation_control_v1')

def prepare():
    config=json.loads(Path('runs/boq_c_night_selected_control_v1/configs/full_C.json').read_text())
    config['root']=str(ROOT/'no_degradation')
    config['finetuning']['gsv_camera_omit']=list(OP_NAMES)
    config['finetuning']['night_runner_sha256']=sha256('reproduction/boq_night_selection.py')
    ROOT.mkdir(parents=True,exist_ok=True)
    seal(ROOT/'config.json',config)
    seal(ROOT/'plan.json',dict(steps_per_trial=9770,distillation=False,omitted_operations=list(OP_NAMES),
         config_sha256=sha256(ROOT/'config.json')))
    return ROOT/'config.json'

def main():
    p=argparse.ArgumentParser();p.add_argument('--wait-for',required=True);a=p.parse_args()
    config=prepare()
    with (ROOT/'.followup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('等待原12组无蒸馏消融；随后无退化对照10轮，最后统一14模型夜间评测。',flush=True)
        wait_for_run(a.wait_for)
        state_path=ROOT/'execution.json'
        if state_path.exists():
            state=json.loads(state_path.read_text())
            if not state.get('id'):raise RuntimeError('Unconfirmed previous launch; inspect Trainroom')
            r=helper('status','--id',state['id'])['run']
            if r['status'] not in ('running','queued','stopping','completed'):
                atomic_json(ROOT/'previous_execution.json',state);state=None
        else:state=None
        if state is None:
            atomic_json(state_path,dict(launch_requested=True))
            state=helper('launch','--cwd',str(Path.cwd()),'--name','AsymVPR · BoQ无退化对照 · 无蒸馏10轮夜间选模',
              '--total-steps','9770','--','/usr/bin/env','OMP_NUM_THREADS=6','OPENBLAS_NUM_THREADS=1',
              'MKL_NUM_THREADS=6','MPLCONFIGDIR=/tmp/asymvpr-mpl','PYTHONDONTWRITEBYTECODE=1',
              '/usr/bin/systemd-run','--user','--scope','--quiet','--unit=boq-no-degradation',
              '-p','MemoryHigh=12G','-p','MemoryMax=18G','-p','MemorySwapMax=0',
              sys.executable,'-u','-m','reproduction.boq_night_selection','--config',str(config),'--arm','control')
            atomic_json(state_path,dict(id=state['id']))
            helper('annotate','--id',state['id'],'--project','AsymVPR','--tags','boq','no-degradation','no-distillation')
        print('No-degradation control:',state['id'],flush=True);wait_for_run(state['id'])
        out=ROOT/'no_degradation/control'
        done=json.loads((out/'completed.json').read_text())
        if done['steps']!=9770 or done['selection']['monitor']!='svox-night-dev/R@1':raise RuntimeError('Incomplete control')
        from reproduction import boq_night_selected_evaluation as evaluation
        original=evaluation.specs
        evaluation.specs=lambda:dict(original(),no_degradation=out/'best.pt')
        sys.argv=[sys.argv[0],'--wait-for',a.wait_for]
        evaluation.main()

if __name__=='__main__':main()
