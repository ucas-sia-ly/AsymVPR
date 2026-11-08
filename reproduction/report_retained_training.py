"""Read completed mixed training histories into a comparison report; never train."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from reproduction.benchmarks import sha256
from reproduction.evaluate import atomic_json


def main():
    out = Path('results/night_backfill_v2'); out.mkdir(parents=True, exist_ok=True)
    registry = json.loads(Path('configs/night_backfill_v2.json').read_text())
    tasks = {'mixed': '539c22d03c8d', 'mixed_camera': '7fe91e0d309a'}
    rows, evidence = {}, {}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for label, task in tasks.items():
        spec = registry['models'][label]
        root = Path(spec['checkpoint']).parent
        done_path = root / 'completed.json'
        done = json.loads(done_path.read_text())
        if done.get('status') != 'completed' or done['steps'] != 9080 or done['epochs'] != 10:
            raise ValueError('Training report requires the completed 10-epoch/9080-step run')
        if done['best']['epoch'] != spec['epoch'] or sha256(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('Training selection differs from fixed evaluation registry')
        log = root / 'logs/metrics.jsonl'
        metrics = [json.loads(s) for s in log.read_text().splitlines()]
        epochs = [json.loads((root / f'epoch_{i:02d}_metrics.json').read_text()) for i in range(1, 11)]
        baseline = json.loads((root / 'baseline.json').read_text())
        title = 'Mixed' if label == 'mixed' else 'Mixed + camera'
        for ax, field in zip(axes[0], ['train/night_asym', 'train/replay_asym']):
            data = [x for x in metrics if field in x]
            ax.plot([x['step'] for x in data], [x[field] for x in data], label=title, linewidth=1.)
        for ax, dataset in zip(axes[1], ['svox_dev', 'msls']):
            scores = [baseline[dataset]['metrics_percent']['R@1']] + [
                x['evaluation'][dataset]['metrics_percent']['R@1'] for x in epochs]
            line, = ax.plot(range(11), scores, marker='.', label=title)
            best = spec['epoch']
            ax.plot(best, scores[best], marker='*', color=line.get_color(), markersize=12)
        selected = epochs[spec['epoch']-1]
        rows[label] = dict(name=spec['name'], status='completed', trainroom_id=task,
            epochs=10, optimizer_updates=9080, night_batch=32, day_replay_batch=32,
            best_epoch=spec['epoch'], best_step=spec['step'], checkpoint_sha256=spec['checkpoint_sha256'],
            svox_night_dev_percent=selected['evaluation']['svox_dev']['metrics_percent'],
            msls_val_percent=selected['evaluation']['msls']['metrics_percent'],
            camera_augmentation=(label == 'mixed_camera'), log=str(log.resolve()))
        evidence[label] = {str(p.resolve()): sha256(p) for p in [done_path, log, root/'baseline.json',
            *[root/f'epoch_{i:02d}_metrics.json' for i in range(1, 11)]]}
    for ax, title in zip(axes.flat, ['Night Eq.8 loss', 'Day replay Eq.8 loss',
                                     'SVOX real-night development R@1', 'MSLS validation R@1']):
        ax.set_title(title); ax.grid(alpha=.25); ax.legend(fontsize=8)
    for ax in axes[0]:
        ax.set_xlabel('Optimizer updates'); ax.set_ylabel('Loss')
    for ax in axes[1]:
        ax.set_xlabel('Epoch'); ax.set_ylabel('Recall@1 (%)')
    image = out / 'mixed_training_comparison.png'
    fig.savefig(image, dpi=160); plt.close(fig)
    atomic_json(out/'training_summary.json', dict(operation='read-only completed-training audit',
        models=rows, evidence_sha256=evidence, plot_sha256=sha256(image)))
    text = ['# Mixed 与 Mixed＋相机退化：已完成训练', '',
        '本次核验历史完整训练记录，不重复训练，也不恢复归档方向。两组从同一原始模型开始，',
        '固定 29,050 张训练夜图、1,529 张地点留出夜图，32 night＋32 day、10 轮、9,080 次更新。', '',
        '| 方法 | 状态 | 最佳 epoch / step | SVOX night dev R@1 | MSLS val R@1 | Trainroom |',
        '| --- | --- | ---: | ---: | ---: | --- |']
    for x in rows.values():
        text.append(f'| {x["name"]} | completed | {x["best_epoch"]} / {x["best_step"]} | '
            f'{x["svox_night_dev_percent"]["R@1"]:.4f} | {x["msls_val_percent"]["R@1"]:.4f} | `{x["trainroom_id"]}` |')
    text += ['', '![真实训练与开发集曲线](mixed_training_comparison.png)', '',
        '星号为实际选中权重；SVOX night dev 用于选模，MSLS 使用原有 1 pp 门槛。',
        'Mixed＋相机退化只对学生夜图施加退化；白天回放、冻结教师目标和测试图片不额外退化。',
        '测试集在选模后统一补测；不能由开发集收益代替各个夜间测试成绩。', '',
        '原始论文复现为 15 轮 / 124,110 更新，使用其既有 epoch 9 best；任务 `223b8a23c9b2` 已完成。',
        '全部曲线来源及完整训练结束标记的 SHA256 见 [training_summary.json](training_summary.json)。', '']
    (out/'TRAINING.md').write_text('\n'.join(text))
    print(out/'TRAINING.md')


if __name__ == '__main__':
    main()
