"""Run the three final-release training controls serially in Trainroom."""
import argparse
import json
from pathlib import Path

from reproduction.augmentation_sweep import run
from reproduction.evaluate import atomic_json, sha256


ROOT = Path('runs/gsv_night_comparison')


def make_plan():
    base = json.loads(Path('configs/boq_b2_gsv_night_final.json').read_text())
    snapshot_path = Path(base['input_dir']) / 'summary.json'
    snapshot = json.loads(snapshot_path.read_text())
    count = int(snapshot['train'])
    steps = ((count + base['night_batch_size'] - 1) // base['night_batch_size']) * base['epochs']
    probe = json.loads((ROOT/'cpu_probe.json').read_text())
    jobs = []
    for tag, name, camera in (('A', '无退化', {}),
                              ('B', '原七项退化', {'gsv_camera': 'legacy'}),
                              ('C', '新版自适应退化', {'gsv_camera': 'adjusted',
                                                 'gsv_edge_threshold': probe['threshold']})):
        config = dict(base, experiment=f'gsv_night_comparison_{tag}',
                      output_dir=str(ROOT/tag/'train'), total_optimizer_steps=steps, **camera)
        path = ROOT/'configs'/f'{tag}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and json.loads(path.read_text()) != config:
            raise ValueError(f'Existing config differs: {path}')
        if not path.exists():
            atomic_json(path, config)
        jobs.append(dict(tag=tag, name=f'AsymVPR · GSV最终版夜间训练 · {tag} {name}',
                         config=str(path), config_sha256=sha256(path), output=config['output_dir']))
    plan = dict(trials=jobs, total_optimizer_steps=len(jobs)*steps,
                optimizer_steps_per_trial=steps, epochs_per_trial=base['epochs'],
                snapshot_sha256=sha256(snapshot_path), probe_sha256=sha256(ROOT/'cpu_probe.json'))
    path = ROOT/'plan.json'
    if path.is_file() and json.loads(path.read_text()) != plan:
        raise ValueError(f'Existing plan differs: {path}')
    if not path.exists():
        atomic_json(path, plan)
    return plan


def report(plan):
    rows = []
    for job in plan['trials']:
        path = Path(job['output'])/'completed.json'
        if path.exists():
            rows.append(dict(tag=job['tag'], best=json.loads(path.read_text())['best'],
                             completion_sha256=sha256(path)))
    atomic_json(ROOT/'training_results.json', dict(rows=rows, complete=len(rows)==len(plan['trials'])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    plan = make_plan()
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.run:
        run(plan, ROOT, report, 'comparison_complete.json')


if __name__ == '__main__':
    main()
