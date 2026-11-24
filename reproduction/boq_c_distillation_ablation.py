"""Serial no-distillation operation omissions, with checkpoint resume."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time

from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json, sha256
from reproduction.gsv_night_camera import OP_NAMES
from reproduction.gsv_night_ablation import OP_ZH

ROOT=Path('runs/boq_dinob_c_ablation_control_v2')
BASE=Path('runs/boq_dinob_c_distillation_v1')


def seal(path,value):
    if path.exists() and json.loads(path.read_text())!=value:
        raise RuntimeError(f'Frozen plan changed: {path}')
    atomic_json(path,value)


def make_plan(root=ROOT):
    root.mkdir(parents=True,exist_ok=True)
    (root/'configs').mkdir(exist_ok=True)
    base=json.loads(Path('configs/boq_dinob_c_distillation.json').read_text())
    jobs=[]
    for operation in OP_NAMES:
        config=dict(base,root=str(root/operation),finetuning=dict(base['finetuning'],gsv_camera_omit=[operation],
                    resume_adapter_sha256=sha256('reproduction/boq_c_ablation_resume.py')))
        path=root/'configs'/f'{operation}.json';seal(path,config)
        for arm in ('control',):
            jobs.append(dict(operation=operation,arm=arm,config=str(path),config_sha256=sha256(path),
                output=str(root/operation/arm),name=f'AsymVPR · BoQ C去除{OP_ZH[operation]} · '+
                ('有蒸馏' if arm=='distill' else '无蒸馏')+' · 10轮'))
    plan=dict(jobs=jobs,epochs=10,steps_per_job=9770,extra_jobs=len(jobs),
        scheduling='wait for completed full-C pair, then serial; fail closed',
        reused_full_C=str(BASE),base_config_sha256=sha256('configs/boq_dinob_c_distillation.json'),
        sources={p:sha256(p) for p in (__file__,'reproduction/boq_c_ablation_resume.py','reproduction/boq_c_distillation.py',
          'reproduction/gsv_night_camera.py','reproduction/boq_protocol.py',
          'reproduction/train_dino_gsv_mixed.py','src/boq.py','src/backbones.py')})
    seal(root/'plan.json',plan)
    return plan


def report(plan):
    rows=[]
    for op in OP_NAMES:
        out=ROOT/op/'control'
        if not (out/'completed.json').exists():continue
        for epoch in range(1,11):
            inputs=[BASE/'control',out]
            data=[json.loads((p/f'epoch_{epoch:02d}.json').read_text()) for p in inputs]
            for dataset in data[0]:
                if len({sha256(p/f'{dataset}_protocol.json') for p in inputs})!=1:
                    raise RuntimeError('Validation protocols differ')
                for metric in ('R@1','R@5','R@10'):
                    rows.append(dict(operation=op,epoch=epoch,dataset=dataset,metric=metric,
                        contribution_pp=data[0][dataset][metric]-data[1][dataset][metric]))
    atomic_json(ROOT/'results.json',dict(rows=rows,complete=len(rows)==len(OP_NAMES)*10*3*3,
        interpretation='No distillation. Full C minus omission, at matched epochs.'))
    lines=['# C版无蒸馏逐项消融','','正值表示完整C优于去除该项；比较同轮结果。','',
           '| 去除项 | 数据集 | 第10轮贡献(pp) |','|---|---|---:|']
    for r in rows:
        if r['epoch']==10 and r['metric']=='R@1':
            lines.append(f"| {OP_ZH[r['operation']]} | {r['dataset']} | {r['contribution_pp']:+.4f} |")
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def wait_for_run(run_id):
    while True:
        run=helper('status','--id',run_id)['run']
        if run['status']=='completed' and run.get('exit_code')==0:return
        if run['status'] not in ('running','queued','stopping'):raise RuntimeError(str(run))
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Reconnect running jobs or relaunch stopped jobs from saved checkpoints')
    args=parser.parse_args()
    plan=make_plan()
    print(json.dumps(dict(extra_jobs=plan['extra_jobs'],epochs=10,steps_per_job=9770),ensure_ascii=False),flush=True)
    if not args.run:return
    with (ROOT/'.queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('等待完整 C 版蒸馏与无蒸馏对照完成；不并发占用 GPU。',flush=True)
        wait_for_run('bef7f894aa81')
        for arm in ('control',):
            done=json.loads((BASE/arm/'completed.json').read_text())
            if done['steps']!=9770:raise RuntimeError('Full C budget differs')
        state_path=ROOT/'execution.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        for job in plan['jobs']:
            if any(sha256(p)!=value for p,value in plan['sources'].items()):raise RuntimeError('Source changed mid-queue')
            if sha256(job['config'])!=job['config_sha256']:raise RuntimeError('Config changed')
            key=job['operation']+'_'+job['arm']
            if key in state and state[key].get('id'):
                prior=helper('status','--id',state[key]['id'])['run']
                if prior['status'] not in ('running','queued','stopping','completed'):
                    if not args.resume:raise RuntimeError('Job interrupted; use --run --resume')
                    state[key]={'previous_ids':state[key].get('previous_ids',[])+[state[key]['id']]}
                    atomic_json(state_path,state)
            if key not in state or ('id' not in state[key] and not state[key].get('launch_requested')):
                history=state.get(key,{}).get('previous_ids',[])
                state[key]={'launch_requested':True,'previous_ids':history};atomic_json(state_path,state)
                response=helper('launch','--cwd',str(Path.cwd()),'--name',job['name'],'--total-steps','9770','--',
                    '/usr/bin/env','OMP_NUM_THREADS=6','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=6',
                    'MPLCONFIGDIR=/tmp/asymvpr-mpl','PYTHONDONTWRITEBYTECODE=1',
                    '/usr/bin/systemd-run','--user','--scope','--quiet',f'--unit=asym-boq-c-omit-{key}',
                    '-p','MemoryHigh=12G','-p','MemoryMax=18G','-p','MemorySwapMax=0',
                    sys.executable,'-u','-m','reproduction.boq_c_ablation_resume',
                    '--config',job['config'],'--arm',job['arm'])
                state[key]={'id':response['id'],'previous_ids':history};atomic_json(state_path,state)
                helper('annotate','--id',response['id'],'--project','AsymVPR','--tags','boq','C-ablation',job['operation'],job['arm'])
            if not state[key].get('id'):raise RuntimeError('Unconfirmed launch; inspect Trainroom')
            print(f"监控 {key}: {state[key]['id']}",flush=True)
            wait_for_run(state[key]['id'])
            done=json.loads((Path(job['output'])/'completed.json').read_text())
            if done['steps']!=9770:raise RuntimeError('Ablation budget differs')
            report(plan)
        atomic_json(ROOT/'completed.json',dict(jobs=len(plan['jobs']),results_sha256=sha256(ROOT/'results.json')))


if __name__=='__main__':main()
