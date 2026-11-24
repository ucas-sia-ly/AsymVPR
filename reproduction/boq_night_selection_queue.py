"""Uniform night-selected full-C and eleven omission trials, all without distillation."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json,sha256
from reproduction.gsv_night_camera import OP_NAMES
from reproduction.gsv_night_ablation import OP_ZH
from reproduction.boq_c_distillation_ablation import wait_for_run,seal

ROOT=Path('runs/boq_c_night_selected_control_v1')


def make_plan():
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'configs').mkdir(exist_ok=True)
    if (ROOT/'plan.json').exists():
        plan=json.loads((ROOT/'plan.json').read_text())
        if any(sha256(p)!=s for p,s in plan['sources'].items()):raise RuntimeError('Source changed')
        for job in plan['jobs']:
            if sha256(job['config'])!=job['config_sha256']:raise RuntimeError('Config changed')
        return plan
    base=json.loads(Path('configs/boq_dinob_c_distillation.json').read_text())
    jobs=[]
    for tag in ('full_C',*OP_NAMES):
        fine=dict(base['finetuning'],selection_monitor='svox-night-dev/R@1',
                  night_runner_sha256=sha256('reproduction/boq_night_selection.py'))
        if tag!='full_C':fine['gsv_camera_omit']=[tag]
        config=dict(base,root=str(ROOT/tag),finetuning=fine)
        path=ROOT/'configs'/f'{tag}.json';seal(path,config)
        jobs.append(dict(tag=tag,config=str(path),output=str(ROOT/tag/'control'),
            config_sha256=sha256(path),name='AsymVPR · BoQ夜间选模 · '+
            ('完整C' if tag=='full_C' else '去除'+OP_ZH[tag])+' · 无蒸馏10轮'))
    plan=dict(jobs=jobs,steps_per_trial=9770,selection='svox-night-dev/R@1',distillation=False,
        sources={p:sha256(p) for p in (__file__,'reproduction/boq_night_selection.py',
              'reproduction/boq_c_ablation_resume.py','reproduction/boq_c_distillation.py',
              'reproduction/boq_protocol.py','src/boq.py','src/backbones.py')})
    seal(ROOT/'plan.json',plan)
    return plan


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    plan=make_plan();print('12 groups, all no-distillation, SVOX-night-dev selection',flush=True)
    if not args.run:return
    with (ROOT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=ROOT/'execution.json';state=json.loads(path.read_text()) if path.exists() else {}
        for job in plan['jobs']:
            if any(sha256(p)!=s for p,s in plan['sources'].items()):raise RuntimeError('Source changed')
            if sha256(job['config'])!=job['config_sha256']:raise RuntimeError('Config changed')
            tag=job['tag'];entry=state.get(tag)
            if entry and entry.get('id'):
                status=helper('status','--id',entry['id'])['run']['status']
                if status not in ('running','queued','stopping','completed'):
                    if not args.resume:raise RuntimeError('Use --run --resume')
                    state[tag]={'history':entry.get('history',[])+[entry['id']]};entry=state[tag]
            if entry is None or ('id' not in entry and not entry.get('launch_requested')):
                history=(entry or {}).get('history',[])
                state[tag]={'launch_requested':True,'history':history};atomic_json(path,state)
                r=helper('launch','--cwd',str(Path.cwd()),'--name',job['name'],'--total-steps','9770','--',
                    '/usr/bin/env','OMP_NUM_THREADS=6','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=6',
                    'MPLCONFIGDIR=/tmp/asymvpr-mpl','PYTHONDONTWRITEBYTECODE=1',
                    '/usr/bin/systemd-run','--user','--scope','--quiet',f'--unit=boq-night-select-{tag}',
                    '-p','MemoryHigh=12G','-p','MemoryMax=18G','-p','MemorySwapMax=0',
                    sys.executable,'-u','-m','reproduction.boq_night_selection','--config',job['config'],'--arm','control')
                state[tag]={'id':r['id'],'history':history};atomic_json(path,state)
                helper('annotate','--id',r['id'],'--project','AsymVPR','--tags','boq','night-selected','no-distillation',tag)
            if not state[tag].get('id'):raise RuntimeError('Unconfirmed launch')
            print(tag,state[tag]['id'],flush=True);wait_for_run(state[tag]['id'])
            done=json.loads((Path(job['output'])/'completed.json').read_text())
            if done['steps']!=9770 or done['selection']['monitor']!='svox-night-dev/R@1':raise RuntimeError('Budget/selection mismatch')
        atomic_json(ROOT/'completed.json',dict(trials=12,selection='svox-night-dev/R@1'))


if __name__=='__main__':main()
