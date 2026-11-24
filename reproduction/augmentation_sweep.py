"""Bounded, serial development-only augmentation screening in Trainroom."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from reproduction.evaluate import atomic_json, sha256

ROOT = Path('runs/augmentation_screen_remaining_v2')
HELPER = '/home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py'


def make_plan():
    base = json.loads(Path('configs/boq_b2_night_v1.json').read_text())
    snapshot = json.loads((ROOT/'inputs/summary.json').read_text())
    steps = (snapshot['train'] + base['night_batch_size'] - 1)//base['night_batch_size']
    derivation = snapshot['derivation']
    base.update(input_dir=str(ROOT/'inputs'), night_teacher_cache=str(ROOT/'cache/night_teacher.npy'),
                replay_excluded_place_ids=derivation['replay_excluded_place_ids'],
                study_input_summary_sha256=sha256(ROOT/'inputs/summary.json'))
    trials = [('none', 0., 1.)]
    trials += [('legacy', p, 1.) for p in [.4, .8, 1.]]
    trials += [('legacy', .8, s) for s in [.5, 1.5]]
    trials += [(kind, .5, s) for kind in ['white_balance', 'local_exposure', 'resolution', 'defocus'] for s in [.5, 1.]]
    trials += [(kind, p, 1.) for kind in ['white_balance', 'local_exposure', 'resolution', 'defocus'] for p in [.25, .75]]
    ROOT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (kind, probability, severity) in enumerate(trials):
        tag = f'{i:02d}_{kind}_p{probability:g}_s{severity:g}'
        config = dict(base, output_dir=str(ROOT/tag/'train'), experiment=tag, epochs=1,
                      total_optimizer_steps=steps, train_num_workers=4, snapshot_workers=2,
                      augmentation_study=dict(version=1, kind=kind, probability=probability, severity=severity))
        path = ROOT/'configs'/f'{tag}.json'
        path.parent.mkdir(exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f'Immutable trial config changed: {path}')
        atomic_json(path, config)
        labels = dict(none='无退化对照', legacy='原七项', white_balance='白平衡',
                      local_exposure='局部曝光', resolution='分辨率损失', defocus='失焦')
        jobs.append(dict(tag=tag, config=str(path), config_sha256=sha256(path),
            output=config['output_dir'], name=f'AsymVPR · 退化筛选 · {labels[kind]} · p={probability:g} s={severity:g}'))
    plan = dict(schema=1, purpose='one-epoch screening, not final selection or full-budget evidence',
                seed=2026, trials=jobs, epochs_per_trial=1, optimizer_steps_per_trial=steps,
                total_optimizer_steps=len(jobs)*steps,
                remaining_snapshot=dict(total=snapshot['total'], train=snapshot['train'], val=snapshot['val'],
                    summary_sha256=base['study_input_summary_sha256']), final_tests=False,
                scheduling='serial; fail closed; resumable existing trial IDs',
                selection='SVOX night development with existing MSLS <=1pp guard',
                limitations=['single seed', 'short horizon; scheduler horizon is one epoch',
                             'probability/severity grid is finite, not a proven optimum',
                             'new operations tested alone, not combined',
                             'no adaptive thresholds during isolated screening'])
    existing = ROOT/'plan.json'
    if existing.exists() and json.loads(existing.read_text()) != plan:
        raise ValueError('Existing sweep plan differs')
    atomic_json(existing, plan)
    return plan


def helper(*args):
    result = subprocess.run(['/usr/bin/python3', HELPER, *args], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout+result.stderr)
    return json.loads(result.stdout)


def summarize(plan):
    rows = []
    for job in plan['trials']:
        out = Path(job['output'])
        path = out/'epoch_01_metrics.json'
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        rows.append(dict(tag=job['tag'], config_sha256=job['config_sha256'],
                         eligible=record['eligible_for_selection'], night_dev_r1=record['score'],
                         msls_delta_pp=record['msls_delta_pp'], completed=(out/'completed.json').exists(),
                         metrics_sha256=sha256(path)))
    atomic_json(ROOT/'screening_results.json', dict(stage='screening_only', rows=rows,
                winner=None, final_tests=False, complete=len(rows)==len(plan['trials'])))
    lines = ['# 退化短程筛选（非最终结论）', '',
        f"每组同一起点、seed=2026、1 epoch / {plan['optimizer_steps_per_trial']} updates。只使用开发集；不运行最终测试。",
        '单种子、短程结果不能证明完整训练收益。新增项先单独测试，不叠加旧七项。', '',
        '| 候选 | SVOX dev R@1 (%) | MSLS Δ (pp) | 满足门槛 |', '|---|---:|---:|---|']
    for r in rows:
        lines.append(f"| {r['tag']} | {r['night_dev_r1']:.4f} | {r['msls_delta_pp']:+.4f} | {r['eligible']} |")
    lines += ['', '后续：对入围项测触发概率与交互；完整10轮、多个种子复核后再冻结最终配置。',
              '当前没有自动选定或替换保留模型。', '']
    (ROOT/'SCREENING.md').write_text('\n'.join(lines))


def run(plan, root=ROOT, report=summarize, completion_name='screening_complete.json',
        retry_failed=(), launch_env=()):
    lock = (root/'.sweep.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Verify the entire frozen input list on CPU before scheduling GPU work.
    from reproduction.night_data import load_records
    load_records(json.loads(Path(plan['trials'][0]['config']).read_text()), 'all')
    report(plan)
    state_path = root/'execution.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'trials': {}}
    for job in plan['trials']:
        if sha256(job['config']) != job['config_sha256']:
            raise RuntimeError('Trial config checksum changed')
        done = Path(job['output'])/'completed.json'
        entry = state['trials'].get(job['tag'])
        if done.exists():
            completion = json.loads(done.read_text())
            if completion['config'] != json.loads(Path(job['config']).read_text()) or completion['steps'] != plan['optimizer_steps_per_trial']:
                raise RuntimeError('Completed trial has different config/budget')
            report(plan)
            continue
        previous = None
        if entry is not None and job['tag'] in retry_failed:
            old_id = entry.get('id')
            if old_id is None:
                raise RuntimeError(f'Cannot retry a launch with unknown task ID: {job["tag"]}')
            old_status = helper('status', '--id', old_id)
            old_task = old_status.get('task', old_status.get('run', old_status))
            if old_task.get('status') == 'failed':
                previous = entry
                entry = None
        if entry is None:
            state['trials'][job['tag']] = dict(launch_requested=True, name=job['name'])
            if previous is not None:
                state['trials'][job['tag']]['previous_attempt'] = previous
            atomic_json(state_path, state)
            response = helper('launch', '--cwd', str(Path.cwd()), '--name', job['name'], '--total-steps', str(plan['optimizer_steps_per_trial']),
                '--', '/usr/bin/env', *launch_env, '/bin/bash', 'reproduction/mixed_camera_launch.sh', '--config', job['config'],
                '--stage', 'train', '--skip-final-evaluation')
            if previous is not None:
                response['previous_attempt'] = previous
            # Save the full small helper response immediately; never relaunch on uncertainty.
            state['trials'][job['tag']] = response
            atomic_json(state_path, state)
            entry = response
            print(json.dumps(dict(event='launched', trial=job['tag'], response=response), ensure_ascii=False), flush=True)
        task_id = entry.get('id') or entry.get('run', {}).get('id')
        if not task_id:
            raise RuntimeError(f'Cannot resolve Trainroom task id: {entry}')
        while True:
            status = helper('status', '--id', task_id)
            task = status.get('task', status.get('run', status))
            phase = task.get('status', status.get('status'))
            if done.exists() and phase == 'completed' and task.get('exit_code') == 0:
                break
            if phase in {'failed', 'completed', 'stopped', 'cancelled', 'error', 'exited'}:
                raise RuntimeError(f'Trial exited without completion: {job["tag"]}: {status}')
            time.sleep(20)
        report(plan)
        print(f'Completed {job["tag"]}; development results updated.', flush=True)
    atomic_json(root/completion_name, dict(status='complete', trials=len(plan['trials']),
                final_tests=False, selected_configuration=None))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = make_plan()
    print(json.dumps(dict(plan=str(ROOT/'plan.json'), trials=len(plan['trials']), steps=plan['total_optimizer_steps'])), flush=True)
    if args.run:
        run(plan)


if __name__ == '__main__':
    main()
