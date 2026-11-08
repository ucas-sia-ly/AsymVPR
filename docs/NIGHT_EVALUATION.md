# 保留模型的多夜间数据集评测

新增补全入口为 `reproduction.night_backfill`，仅包含同样三个保留模型，覆盖 11 个协议、33 项方法×协议。
新增 RobotCar-Seasons public Night/Night-rain、Gardens Point 两种白天图库和 SPED 混合光照诊断。
见[补全协议](NIGHT_BACKFILL.md)、[完整训练曲线](../results/night_backfill_v2/TRAINING.md)和
[最新统一报告](../results/night_backfill_v2/REPORT.md)。下文保留此前六组协议的原始记录。

入口：`python -m reproduction.night_suite`。只评测原始、混合、混合＋退化三个固定 best 权重，不训练、不重新选模、不恢复已放弃的方法。

| 协议 | 图库 | Query | 正例与边界 |
| --- | ---: | ---: | --- |
| MSLS val 夜间→白天 | 18,841 | 55 | 原始 night 标签选夜间 query、白天 gallery；排除 panorama，同城 UTM ≤25m；自定义子集 |
| MSLS test 夜间→白天 | 36,362 | 309 | 同上，使用本机公开元数据；不是官方完整 all/subtask 分数 |
| SVOX test-night | 17,166 | 823 | test 夜间查询，UTM ≤25m；与 train-night 选模开发集分开 |
| DarkDriving 配对诊断 | 9,530 | 9,530 | 同名 Day/Night 配对才算正例，邻近帧可能成为假负例；不等同地点半径 Recall |
| RobotCar GPS 夜间诊断 | 21,006 | 22,495 | 三个白天 traversal 图库、三个夜间 traversal 查询，已有 1m GPS 采样精确时间戳关联，UTM ≤25m；561 个无正例查询显式排除 |
| Tokyo105 night | 完整 Tokyo 图库 | 105 | 核实既有封存结果与权重后汇总，不重新推理 |

不将缺失检索真值的 Aachen 官方夜间 test、RobotCar-Seasons 官方 test、NightCity 等图片目录编造为可报告的 Recall。

所有推理使用原始图片，ToTensor → bicubic Resize（antialias）→ ImageNet Normalize，FP32、TF32 关闭、精确内积检索。混合＋退化模型只在之前的训练阶段对学生夜图使用了退化，本次测试没有额外退化。

图库特征及混合模型查询特征可从已归档的同协议评测缓存复用。复用前重建并严格比较图片顺序、文件状态、真值、元数据、权重、预处理和完整特征 SHA256。复用数据通过硬链接纳入当前评测目录，所有搜索只读；元数据独立写入，不修改归档文件。`feature_io.py` 只提供有界只读特征访问。

新查询特征逐批提交进度；中断后相同配置可恢复。资源按启动时实际空闲显存预留 4GiB，当前评测进程 CUDA 分配预算最多 8GiB。当前批量 8、worker 1，主机 MemoryHigh=4GiB、MemoryMax=5GiB、SwapMax=0。不影响其他项目任务。

结果随每个模型/数据集完成后写入 [报告](../results/mixed_night_suite_v1/REPORT.md)和[完整汇总](../results/mixed_night_suite_v1/summary.json)。每条结果包含权重、协议、预测及特征校验和，并保存每张 query 的命中标记与 top-10。Trainroom 使用真实提取图数和评测阶段，指标为 `eval/<协议>/<模型>/R@K`（百分数），不是训练更新。

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR --name AsymVPR-mixed-night-suite-v1 \
  -- /usr/bin/env ASYMVPR_EVAL_MODULE=reproduction.night_suite \
  ASYMVPR_EVAL_MEMORY_HIGH=4G ASYMVPR_EVAL_MEMORY_MAX=5G \
  /bin/bash reproduction/eval_launch.sh --batch-size 8 --workers 1
```

启动前检查同一任务是否正在运行；已有任务应接入而不是重复启动。完整完成后生成 `complete.json`，绑定汇总与报告的 SHA256。
