"""Run only the authorized paired C-policy distillation comparison, serially."""
import fcntl
import json
from pathlib import Path
import sys
import time
from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json, sha256


def main():
    root=Path('runs/boq_dinob_c_distillation_v1');root.mkdir(parents=True,exist_ok=True)
    with (root/'.pair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=root/'execution.json'
        state=json.loads(path.read_text()) if path.exists() else {}
        sources={p:sha256(p) for p in ('configs/boq_dinob_c_distillation.json',
                  'reproduction/boq_c_distillation.py','reproduction/boq_c_distillation_pair.py')}
        if state.get('sources',sources)!=sources:raise RuntimeError('Pair source changed')
        state['sources']=sources
        for arm in ('distill','control'):
            if arm not in state:
                state[arm]={'launch_requested':True};atomic_json(path,state)
                label='C退化＋蒸馏' if arm=='distill' else 'C退化无蒸馏对照'
                response=helper('launch','--cwd',str(Path.cwd()),'--name',
                    f'AsymVPR · DINOv2-BoQ · {label} · 8比1微调10轮','--total-steps','9770','--',
                    '/usr/bin/env','OMP_NUM_THREADS=6','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=6',
                    'MPLCONFIGDIR=/tmp/asymvpr-mpl','PYTHONDONTWRITEBYTECODE=1',
                    '/usr/bin/systemd-run','--user','--scope','--quiet',f'--unit=asym-boq-c-{arm}-v1',
                    '-p','MemoryHigh=12G','-p','MemoryMax=18G','-p','MemorySwapMax=0',
                    sys.executable,'-u','-m','reproduction.boq_c_distillation','--arm',arm)
                state[arm]={'id':response['id']};atomic_json(path,state)
                helper('annotate','--id',response['id'],'--project','AsymVPR','--tags','boq','C-policy',arm)
            if not state[arm].get('id'):raise RuntimeError('Unconfirmed launch; inspect Trainroom')
            while True:
                run=helper('status','--id',state[arm]['id'])['run']
                if run['status']=='completed' and run.get('exit_code')==0:break
                if run['status'] not in ('running','queued','stopping'):raise RuntimeError(str(run))
                time.sleep(20)
            done=json.loads((root/arm/'completed.json').read_text())
            if done['steps']!=9770:raise RuntimeError('Incorrect update budget')
        atomic_json(root/'comparison.json',{arm:{'baseline':json.loads((root/arm/'baseline.json').read_text()),
            'epochs':[json.loads((root/arm/f'epoch_{i:02d}.json').read_text()) for i in range(1,11)]}
            for arm in ('distill','control')})


if __name__=='__main__':main()
