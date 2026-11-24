"""Dispatch the equal-budget GSV ratio trials serially through Trainroom."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time

from reproduction.augmentation_sweep import helper
from reproduction.evaluate import atomic_json, sha256
from reproduction.train_dino_gsv_mixed import DEFAULT_CONFIG, MixedViews, load_config, trial_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=DEFAULT_CONFIG)
    parser.add_argument('--preflight', action='store_true', help='Print CPU batch plans without launching training')
    args = parser.parse_args()
    config = load_config(args.config)
    data = MixedViews(config)
    total = data.steps*config['epochs']
    if args.preflight:
        print(json.dumps([trial_summary(data, config, ratio) for ratio in config['ratios']], indent=2))
        return
    root = Path(config['root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.sweep.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch(data, config, args.config, total, root)


def dispatch(data, config, config_path, total, root):
    trials = []
    for ratio in config['ratios']:
        tag = ratio.replace(':', 'to')
        trials.append(dict(trial_summary(data, config, ratio), tag=tag, output=str(root/tag)))
    plan = dict(config=config, steps_per_trial=total, steps_per_epoch=data.steps,
                train_places=len(data.places), trials=trials,
                source_sha256={name: sha256(name) for name in
                               (config_path, config['base_config'], 'reproduction/train_dino_gsv_mixed.py',
                                'reproduction/boq_protocol.py', 'src/boq.py', 'src/backbones.py')})
    plan_path = root/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('The frozen sweep plan changed')
    atomic_json(plan_path, plan)
    state_path = root/'execution.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    active = None

    def stop(*_):
        if active:
            phase = helper('status', '--id', active)['run']['status']
            if phase in {'running', 'stopping'}:
                helper('stop', '--id', active)
        raise SystemExit(130)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    results = []
    for trial in trials:
        ratio, tag, out = trial['ratio'], trial['tag'], Path(trial['output'])
        if any(sha256(name) != checksum for name, checksum in plan['source_sha256'].items()):
            raise RuntimeError('Config or training source changed mid-sweep')
        entry = state.get(tag)
        if entry is None:
            label = '纯真实' if ratio == 'real' else f'真实:生成 {ratio}'
            name = f'AsymVPR · DINOv2-B + BoQ 混合训练 · {label} · {config["epochs"]}轮'
            state[tag] = dict(launch_requested=True, name=name)
            atomic_json(state_path, state)
            response = helper('launch', '--cwd', str(Path.cwd()), '--name', name,
                '--total-steps', str(total), '--', '/usr/bin/env',
                f'OMP_NUM_THREADS={config["cpu_threads"]}', 'OPENBLAS_NUM_THREADS=1',
                f'MKL_NUM_THREADS={config["cpu_threads"]}', '/usr/bin/systemd-run',
                '--user', '--scope', '--quiet', f'--unit=asym-dino-gsv-mix-{tag}-{os.getpid()}',
                '-p', 'MemoryHigh=10G', '-p', 'MemoryMax=13G', '-p', 'MemorySwapMax=0',
                sys.executable, '-u', '-m', 'reproduction.train_dino_gsv_mixed',
                '--config', config_path, '--ratio', ratio)
            entry = state[tag] = dict(id=response['id'], name=name)
            atomic_json(state_path, state)
            print(f'Launched {ratio}: {entry["id"]}', flush=True)
            active = entry['id']
            try:
                helper('annotate', '--id', active, '--project', 'AsymVPR',
                       '--tags', 'dino-b', 'boq', 'gsv-mixed', tag)
            except (RuntimeError, OSError) as exc:
                print(f'Could not label {active}: {exc}', flush=True)
        if not entry.get('id'):
            raise RuntimeError(f'Unconfirmed Trainroom launch for {ratio}; check before starting another')
        active = entry['id']
        while True:
            run = helper('status', '--id', active)['run']
            phase = run['status']
            if phase == 'completed' and run.get('exit_code') == 0:
                break
            if phase not in {'running', 'stopping', 'queued'}:
                raise RuntimeError(f'Trainroom trial {active} is {phase}: {run.get("log_tail", "")}')
            time.sleep(20)
        complete = json.loads((out/'completed.json').read_text())
        if complete['steps'] != total or complete['identity']['config']['ratio'] != ratio:
            raise RuntimeError(f'Completed trial does not match the frozen budget: {ratio}')
        epochs = [json.loads((out/f'epoch_{epoch:02d}.json').read_text())
                  for epoch in range(1, config['epochs']+1)]
        results.append(dict(ratio=ratio, trainroom_id=active, per_epoch=epochs))
        atomic_json(root/'comparison.json', dict(complete=len(results)==len(trials), results=results))
        print(f'Completed {ratio}: {epochs[-1]}', flush=True)
        active = None


if __name__ == '__main__':
    main()
