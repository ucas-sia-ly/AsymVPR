"""Paired omissions of individual operations and families on the final GSV release."""
import argparse
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from reproduction.augmentation_sweep import run
from reproduction.evaluate import atomic_json, sha256
from reproduction.gsv_night_camera import FAMILIES, OP_NAMES, RANGES
from reproduction.night_eval import load_svox_dev


ROOT = Path('runs/gsv_night_ablation')
CONTROL = Path('runs/gsv_night_comparison')
SCREEN_EPOCHS = 2
FAMILY_NAMES = ('color', 'detail', 'exposure')
OP_ZH = dict(saturation='降饱和', white_balance='白平衡', defocus='失焦', resolution='降分辨率',
             motion_blur='运动模糊', noise='噪声', jpeg='JPEG', gain_gamma='增益/Gamma',
             local_exposure='局部曝光', vignette='暗角', bloom='泛光',
             color='颜色族', detail='细节族', exposure='曝光族')


def read(path):
    return json.loads(path.read_text())


def write_sealed(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing ablation input differs: {path}')
    else:
        atomic_json(path, value)


def control_config(group):
    path = CONTROL/group/'train/completed.json'
    completed = read(path)
    if completed['status'] != 'completed' or completed['epochs'] != 10:
        raise ValueError(f'Expected a completed ten-epoch control: {path}')
    return completed['config']


def make_plan(root, specifications, epochs):
    root.mkdir(parents=True, exist_ok=True)
    controls = {group: control_config(group) for group in ('B','C')}
    snapshot_path = Path(controls['B']['input_dir'])/'summary.json'
    snapshot = read(snapshot_path)
    size = controls['B']['night_batch_size']
    steps = ((snapshot['train'] + size - 1)//size)*epochs
    jobs = []
    for group, name, omission in specifications:
        tag = f'{group}_minus_{name}'
        out = root/tag/'train'
        base = controls[group]
        config = dict(base, experiment=f'gsv_night_ablation_{epochs}ep_{tag}', output_dir=str(out),
                      epochs=epochs, schedule_epochs=base['epochs'], total_optimizer_steps=steps,
                      gsv_camera_omit=list(omission))
        path = root/'configs'/f'{tag}.json'
        path.parent.mkdir(exist_ok=True)
        write_sealed(path, config)
        jobs.append(dict(tag=tag, group=group, removed=list(omission),
                         name=f'AsymVPR · GSV退化消融 · {group} 去除{OP_ZH[name]} · {epochs}轮',
                         config=str(path), config_sha256=sha256(path), output=str(out)))
    plan = dict(trials=jobs, epochs=epochs, optimizer_steps_per_trial=steps,
                total_optimizer_steps=steps*len(jobs), snapshot_sha256=sha256(snapshot_path),
                controls={group: sha256(CONTROL/group/'train/completed.json') for group in 'ABC'})
    write_sealed(root/'plan.json', plan)
    return plan


@lru_cache(maxsize=1)
def positives():
    return load_svox_dev(control_config('B'))['positives']


def compare(out, control, epoch):
    path = out/f'epoch_{epoch:02d}_metrics.json'
    base_path = control/path.name
    row, base = read(path), read(base_path)
    a, b = row['evaluation']['svox_dev'], base['evaluation']['svox_dev']
    if a['definition'] != b['definition']:
        raise ValueError('SVOX retrieval protocols differ')
    pred_path = f'evaluation/svox_night_dev/epoch_{epoch:02d}_top10.npy'
    pred, base_pred = (np.load(p/pred_path, allow_pickle=False) for p in (out,control))
    delta = {}
    for k in (1,5,10):
        candidate_hit = np.array([np.isin(p[:k], truth).any() for p,truth in zip(pred, positives(), strict=True)])
        base_hit = np.array([np.isin(p[:k], truth).any() for p,truth in zip(base_pred, positives(), strict=True)])
        diff = candidate_hit.astype(np.int8) - base_hit
        if not np.isclose(diff.mean()*100, a['metrics_percent'][f'R@{k}']-b['metrics_percent'][f'R@{k}']):
            raise ValueError('Saved SVOX predictions disagree with reported recall')
        delta[f'R@{k}'] = dict(delta_pp=float(diff.mean()*100), improved=int((diff>0).sum()),
                               regressed=int((diff<0).sum()))
    return dict(epoch=epoch, svox_r1=row['score'], control_svox_r1=base['score'],
                msls_delta_pp=row['msls_delta_pp'], passes_msls_guard=row['eligible_for_selection'],
                retrieval_delta=delta)


def report(root, plan):
    rows = []
    epochs = plan['epochs']
    for job in plan['trials']:
        out = Path(job['output'])
        completed = out/'completed.json'
        if completed.exists():
            control = CONTROL/job['group']/'train'
            entry = dict(tag=job['tag'], group=job['group'], removed=job['removed'],
                         endpoints=[compare(out,control,epoch) for epoch in (1,epochs)],
                         selected_best=read(completed)['best'], control_best=read(control/'completed.json')['best'])
            rows.append(entry)
    atomic_json(root/'results.json', dict(complete=len(rows)==len(plan['trials']), rows=rows))
    lines = ['# GSV 最终版夜间退化消融', '',
             f'已完成 {len(rows)}/{len(plan["trials"])} 组。差值定义为「去除后减完整策略」。',
             '负值支持被去除操作有贡献。所有行按同轮比较，使用同一 SVOX night dev 和原 MSLS 门槛。',
             '筛查采用同一训练种子，罕见操作的结果可能波动。', '',
             '| 组别 | 去除 | 第1轮 Δ R@1 | 终轮 Δ R@1 | 终轮新增命中 / 丢失命中 | MSLS Δ |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        early, end = row['endpoints']
        a, b = early['retrieval_delta']['R@1'], end['retrieval_delta']['R@1']
        removed = '、'.join(OP_ZH[name] for name in row['removed'])
        lines.append(f"| {row['group']} | {removed} | {a['delta_pp']:+.2f} | {b['delta_pp']:+.2f} | "
                     f"{b['improved']} / {b['regressed']} | {end['msls_delta_pp']:+.2f} |")
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def confirmation_specs(results):
    """Confirm strongest increase and decrease, with directional agreement across epochs."""
    by_direction = {}
    for direction in (1,-1):
        eligible = []
        for row in results:
            first, last = row['endpoints']
            a, b = (direction*end['retrieval_delta']['R@1']['delta_pp'] for end in (first,last))
            if a > 0 and b >= .50 and last['passes_msls_guard']:
                eligible.append((b,a,row['tag'],row))
        if eligible:
            row = max(eligible)[-1]
            by_direction[direction] = (row['group'],row['tag'].split('_minus_',1)[1],row['removed'])
    return list(by_direction.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--retry-failed', action='append', default=[], metavar='TRIAL')
    parser.add_argument('--memory-high', choices=('8G', '9G', '10G'), default='8G')
    args = parser.parse_args()
    specs = [('B', name, (name,)) for name in (*RANGES, 'saturation')]
    specs += [('C', name, (name,)) for name in OP_NAMES]
    specs += [('C', name, tuple(ops)) for name, (_,ops) in zip(FAMILY_NAMES,FAMILIES)]
    plan = make_plan(ROOT,specs,SCREEN_EPOCHS)
    print(f'{len(specs)} paired omissions; {plan["total_optimizer_steps"]} screening updates.',flush=True)
    if args.run:
        launch_env = (f'ASYMVPR_TRAIN_MEMORY_HIGH={args.memory_high}',)
        unknown = set(args.retry_failed) - {job['tag'] for job in plan['trials']}
        if unknown:
            parser.error(f'Unknown retry trial: {sorted(unknown)}')
        run(plan,ROOT,lambda p: report(ROOT,p), retry_failed=args.retry_failed, launch_env=launch_env)
        followups = confirmation_specs(read(ROOT/'results.json')['rows'])
        atomic_json(ROOT/'confirmation_selection.json',dict(selected=followups, minimum_epoch2_delta_pp=.50,
                    require_same_direction_epoch1=True, require_msls_guard=True))
        if followups:
            confirm_root = ROOT/'confirm'
            full = make_plan(confirm_root,followups,10)
            run(full,confirm_root,lambda p:report(confirm_root,p), 'confirmation_complete.json',
                launch_env=launch_env)


if __name__ == '__main__':
    main()
