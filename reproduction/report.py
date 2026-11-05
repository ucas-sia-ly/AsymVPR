"""Generate a small auditable report from actual recorded results only."""
import argparse
import json
from pathlib import Path


def generate(summary_path='results/evaluation/summary.json', output='results/REPORT.md'):
    root = Path(__file__).resolve().parents[1]
    history = json.loads((root / 'results/training_history.json').read_text())
    summary_path = Path(summary_path)
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {'datasets': {}}
    execution_path = summary_path.parent / 'execution.json'
    execution = json.loads(execution_path.read_text()) if execution_path.exists() else None
    discarded_summary = False
    if execution and 'checkpoint_sha256' in execution and execution['checkpoint_sha256'] != summary.get('checkpoint_sha256'):
        discarded_summary = bool(summary.get('datasets'))
        summary = {'datasets': {}}
    checkpoint_sha = summary.get('checkpoint_sha256')
    expected_checkpoint = (execution['checkpoint_sha256']
                           if execution and 'checkpoint_sha256' in execution
                           else checkpoint_sha or history['best_checkpoint']['sha256'])
    is_historical_best = checkpoint_sha == history['best_checkpoint']['sha256']
    labels = {'msls': 'MSLS val', 'pitts250k': 'Pitts250k test', 'tokyo247': 'Tokyo24/7',
              'nordland': 'Nordland', 'amstertime': 'AmsterTime'}
    lines = ['# AsymVPR 主配置复现报告', '',
             'DINOv2-B + BoQ 图库 → EfficientViT-B2 + BoQ 查询。独立实现；作者仓库本次核对返回 404。', '',
             '## 训练来源', '',
             '复用本机先前完成的 15 epochs、124,110 steps 训练。本次没有重复训练。',
             '最佳权重由 MSLS validation R@1 选择：epoch 9、step 74,466。',
             f'权重 SHA256：`{history["best_checkpoint"]["sha256"]}`。', '',
             '历史训练最佳 MSLS R@1/5/10：92.1622 / 96.2162 / 96.3514%。',
             '下表仅使用本次独立评测输出；空项表示尚未产生结果。', '',
             '## 独立评测', '',
             '| 数据集 | 本次 R@1 % | 论文 R@1 % | 差值（百分点） | 本次 R@5 % |',
             '| --- | ---: | ---: | ---: | ---: |']
    if checkpoint_sha and not is_historical_best:
        lines[lines.index('## 独立评测'):lines.index('## 独立评测')] = [
            f'**本次评测使用另一权重**：`{checkpoint_sha}`；上面的 epoch 9 记录仅说明历史训练来源。', '']
    if execution:
        state = f'本次评测进程状态：**{execution["status"]}**。尚无结果的数据集保持空缺。'
        if execution.get('error'):
            state += f' 进程错误：`{execution["error"]}`。'
        if discarded_summary:
            state += ' 已排除属于其他权重的旧汇总，未将其当作本次成绩。'
        lines.insert(lines.index('## 独立评测') + 2, state + '\n')
    for name, label in labels.items():
        value = summary.get('datasets', {}).get(name)
        if value:
            metrics, paper = value['metrics_percent'], value['paper_table1_percent']
            delta = f'{metrics["R@1"]-paper["R@1"]:+.4f}'
            if value.get('paper_comparison_note'):
                delta = '—（协议不同）'
            lines.append(f'| {label} | {metrics["R@1"]:.4f} | {paper["R@1"]:.1f} | {delta} | {metrics["R@5"]:.4f} |')
        else:
            lines.append(f'| {label} | 尚未产生 | — | — | — |')
    lines += ['', 'Gallery 始终由冻结教师编码，query 由同一个最佳学生权重编码；FP32、TF32 关闭，分块穷举检索。',
              'MSLS 使用 740 个有真值查询；它同时是选模验证集，不能视作独立 test 成绩。', '',
              '### 实际评测协议', '']
    for name, value in summary.get('datasets', {}).items():
        lines += [f'- **{labels[name]}**：{value["references"]["count"]:,} references / {value["queries"]["count"]:,} queries；`{json.dumps(value["protocol"], ensure_ascii=False)}`。']
        if value.get('paper_comparison_note'):
            lines += [f'  {value["paper_comparison_note"]}。']
    diagnostic_path = root / 'results/nordland_protocol_diagnostic.json'
    nordland = summary.get('datasets', {}).get('nordland', {})
    diagnostic_record = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else {}
    if (nordland.get('protocol', {}).get('protocol_variant') == 'paper'
            and diagnostic_record.get('checkpoint_sha256') == checkpoint_sha
            and diagnostic_record.get('predictions_sha256') == nordland.get('predictions_sha256')):
        diagnostic = diagnostic_record['metrics_percent']
        lines += ['', f'Nordland 协议诊断：对同一组预测改用发布的 ±1 帧真值，R@1/5 为 '
                  f'{diagnostic["R@1"]:.4f} / {diagnostic["R@5"]:.4f}%。'
                  '主表按论文文字采用 ±3 帧；容差对分数影响很大，不能把高于论文的差值直接解释为方法改进。']
    eff_path = root / 'results/efficiency.json'
    if eff_path.exists():
        eff = json.loads(eff_path.read_text())
        if eff['checkpoint_sha256'] != expected_checkpoint:
            eff = None
    else:
        eff = None
    if eff:
        lines += ['', '## 本机推理计时', '',
                  '| 模型 | 参数量 | batch=1 中位延迟 ms |', '| --- | ---: | ---: |']
        for role, label in [('gallery', 'DINOv2-B + BoQ'), ('query', 'EfficientViT-B2 + BoQ')]:
            record = eff['models'][role]
            lines.append(f'| {label} | {record["parameters"]:,} | {record["latency_ms_median"]:.3f} |')
        lines += ['', f'同机网络前向中位延迟比为 {eff["query_speedup_median"]:.2f}×。',
                  f'{eff["gpu"]}，FP32、322×322，预热 {eff["warmup"]} 次，计时 {eff["iterations"]} 次，CUDA 同步 wall clock。',
                  '计时使用重复的合成 GPU 输入，不含读图、预处理、传输或检索；不能直接与论文的硬件计时比较。']
    lines += ['', '## 复现边界', '',
              '- 本机 GSV-Cities 为 529,506 图像、64,394 地点；论文写约 560k / 67k。',
              '- 论文未公开完整增强、插值、采样、协方差估计和 checkpoint 选择细节；实际选择详见 README。',
              '- Tokyo 使用本机原始 VGA PNG，AmsterTime 使用原始现代 PNG / 历史 JPG；作者评测图像编码未公开。',
              '- Nordland 默认使用 VPR-Bench 发布，冬季 references、夏季 queries，query 每 10 帧采样；具体正匹配容差见实际评测协议。论文写 ±3 帧，但没有明确季节方向。',
              '- 尚未复现 SALAD、MobileViTv2、全部对比方法及消融实验。', '',
              '有序图像清单的哈希、元数据/真值/模型/特征哈希和预测 Top10 均保存在 evaluation 目录。',
              '训练完整历史来源见 training_history.json；训练和计时均保留 Trainroom 原始日志。', '']
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines))
    print(f'Report written: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', default='results/evaluation/summary.json')
    parser.add_argument('--output', default='results/REPORT.md')
    args = parser.parse_args()
    generate(args.summary, args.output)


if __name__ == '__main__':
    main()
