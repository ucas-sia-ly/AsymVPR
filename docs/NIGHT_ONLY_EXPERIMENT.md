# BoQ → EfficientViT-B2 night-only 对照实验

本文记录新实验 `boq_b2_night_only_v1` 的协议和执行入口。新实验的训练结果与实际运行状态以 Trainroom 及其输出文件为准；本文不预先填写新成绩。入口为 `python -m reproduction.night_only_train`，配置为 `configs/boq_b2_night_only_v1.json`，输出根目录为 `runs/boq_b2_night_only_v1/`。

初始化继续使用原始白天复现的 `checkpoints/boq_b2_best.pt`，即 epoch 9、step 74,466，与此前混合训练的初始化一致。原始文件 SHA256 为 `d051f6cec174614a7fed1391624420308f31cac6aa59782530f582a01eacb54a`。此前混合实验选出的 epoch 7 权重用于结果对照，不作为本轮初始化。

本轮复用已经封存的 30,579 张生成夜图、精确 GSV 源图映射、标签和 95%/5% 地点划分：29,050 张训练夜图、1,529 张合成开发夜图。原快照位于 `runs/boq_b2_night_v1/inputs/`，`records.jsonl` SHA256 为 `5bad1c9e167aba58c0ab177f494902785cb30f1a999c13743e327cfcf63f31cb`。来自 `night_place_fast_v1` 与 `night_cluster_v1` 的后续新增图片不自动加入本轮。需要扩大数据时另建实验配置和输入/输出目录。

每个地点和源图只属于本轮的一个划分。初始化模型已训练过相应 GSV 白天地点，因此合成开发集是新增夜间外观的留出集。生成图的来源 SHA256、完整解码、尺寸、精确配对和标签已在快照阶段核对；原始 `pending_verification`、`passed=false`、`eligible_for_training=false` 等状态继续保留。用户授权本轮使用这些图，不会把上游状态改写成语义审核通过。

与此前混合训练的具体差异为：

| 项目 | 已完成的 mixed v1 | 新 night-only v1 |
| --- | --- | --- |
| 每个正常学生 batch | 32 night + 32 global day replay | 32 night |
| 学生训练前向与 BatchNorm 统计 | 夜图与白天回放共同参与 | 只包含训练夜图 |
| 夜图正目标 | 按 day/night 教师一致性融合，源白天系数至少 0.5 | 纯冻结 `T_night` 描述子 |
| 白天回放损失 | 与夜图损失等权平均 | 无白天回放分支 |
| 源白天教师特征 | 参与夜图目标与诊断 | 仅用于诊断，不进入正目标或损失 |
| 全局负地点中心 / 对角协方差 | 原冻结白天银行 | 继续保留原冻结白天银行 |
| 最佳 checkpoint 选择 | 最大 SVOX night dev R@1，且 MSLS 相对初始下降 ≤1 pp | 最大 SVOX night dev R@1；MSLS 只监控 |

固定教师始终处于推理模式。白天图库、源白天教师诊断和开发评测不会向学生训练前向加入白天图片，也不会通过这些诊断更新参数。保留原始白天地点银行，使本轮描述子继续与已有 BoQ 图库空间兼容；“night-only”指学生优化所见的图片和夜图正目标。

夜图查询 `q_night` 的正目标直接使用归一化冻结教师夜图描述子 `T_night`。训练损失继续采用 Eq. (8)，地点 ID 来自精确源图的原全局标签；使用除自身地点外的全部负地点中心、固定对角协方差、`tau=0.05`、`gamma=15`、4,096 地点分块。没有 teacher-consistency gate、强制 source-day 混合或额外 replay loss。

| 预算 | mixed v1 | night-only v1 |
| --- | ---: | ---: |
| epochs | 10 | 10 |
| 每 epoch 夜图 | 29,050 | 29,050 |
| 每 epoch optimizer updates | 908 | 908 |
| optimizer updates 总数 | 9,080 | 9,080 |
| 夜图暴露总数 | 290,500 | 290,500 |
| 学生训练图像视图总数 | 581,000 | 290,500 |

两实验每个 epoch 都完整遍历训练夜图一次，正常 night batch 为 32，尾 batch 为 26。对照匹配夜图数量和更新次数；学生处理的总图片数及 BatchNorm 输入分布随移除回放而改变。原混合实验还使用不同目标和选模门槛，因此最终差异不能单独归因于其中某一个因素。

开发评测继续采用同一冻结教师图库、FP32、关闭 TF32 和精确检索，记录以下 Recall@1/5/10，单位为百分数：

- 合成开发：1,529 张夜图查询，30,579 个冻结来源的白天教师描述子作为图库；同地点为正例。它用于诊断新增外观适配。
- MSLS val：740 queries / 18,871 references，官方 25 m positives。它展示白天检索能力变化，本轮不以其跌幅排除 checkpoint。
- SVOX night dev：641 个真实夜间查询 / 22,232 张完整 train gallery，25 m UTM 正例。开发查询已排除距离任何 SVOX val/test 图库或查询图小于 50 m 的位置；范围是 Oxford 同城、同相机目标域开发数据。

本轮按 SVOX night dev R@1 选择最佳权重，初始化基线也保留作比较。Tokyo24/7 的固定 105 张夜间查询用于最终测试，不参与 epoch 选择、学习率调整或提前停止。完整 Tokyo315 和其他固定测试也在选模完成后报告。SVOX 开发成绩包含选模作用，不能当作未见域测试成绩。

此前 mixed v1 已完成 10 epochs / 9,080 updates，选中 **epoch 7 / step 6,356**。其 best SHA256 为 `c47c3ee95f6f6a92a6823cd4c98e2de2e1d8e4191e11ec735c12dbe9278459d2`，来源为 `runs/boq_b2_night_v1/train/best.pt`。新实验应与这个选中权重比较。旧实验的最后一轮 SVOX R@1 为 89.0796%，低于其选中轮次。

| R@1，% | 原始初始化 | mixed v1 best | mixed 相对初始化 Δ pp | night-only v1 |
| --- | ---: | ---: | ---: | --- |
| SVOX night dev | 80.4992 | 89.5476 | +9.0484 | 待评测 |
| MSLS val | 92.1622 | 91.7568 | −0.4054 | 待评测 |
| Tokyo night，105 queries | 83.8095 | 87.6190 | +3.8095 | 待最终评测 |
| Tokyo 全部 315 queries | 91.7460 | 93.3333 | +1.5873 | 待最终评测 |
| Pitts250k test | 94.7947 | 94.8309 | +0.0362 | 待最终评测 |
| Nordland，±3 帧 | 78.5145 | 79.5290 | +1.0145 | 待最终评测 |
| AmsterTime | 47.6036 | 49.3095 | +1.7059 | 待最终评测 |

mixed 最佳权重的 SVOX R@5/R@10 为 **94.6958 / 95.6318**；Tokyo night 为 **94.2857 / 94.2857**。Tokyo night 的 R@1 从 88/105 变为 92/105，包含修好 5、改坏 1；该诊断没有参与旧实验选模。完整旧结果和协议见 [`runs/boq_b2_night_v1/NIGHT_REPORT.md`](../runs/boq_b2_night_v1/NIGHT_REPORT.md) 与 [`final_evaluation/comparison.json`](../runs/boq_b2_night_v1/final_evaluation/comparison.json)。

本机训练通过 Trainroom 接入真实日志。已有对应任务时优先查看或恢复该任务；启动新进程前确认同一输出目录的旧进程已经退出。计划执行入口为：

```bash
cd /home/admin123/github/AsymVPR
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py ensure
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR \
  --name 'AsymVPR BoQ-B2 night-only v1' --total-steps 9080 \
  -- /bin/bash reproduction/night_only_launch.sh \
  --config configs/boq_b2_night_only_v1.json --stage all
```

恢复时使用同一配置和输出目录，将入口阶段设为 `--stage train`，从本轮 `train/last.pt` 继续。准备阶段验证并复用固定快照、夜图教师缓存和评测缓存；这些准备计数不代表 optimizer updates。训练日志记录真实 global step、night loss、教师/学生余弦诊断、梯度范数、分组学习率、吞吐和显存；开发指标分别区分 synthetic、MSLS 和 SVOX night。完整状态、epoch 权重、最佳权重及完成标记保存在新实验目录，原 mixed v1 的训练状态与报告保留。

`night_only_launch.sh` 使用已有 `boq` Python 环境，systemd scope 设置 MemoryHigh=8 GiB、MemoryMax=10 GiB、MemorySwapMax=0；磁盘 mmap 复用大特征数组。GPU 分配预算按启动时的实际空闲显存设置并保留余量。Trainroom 可查看日志与关键指标，语义审核尚未完成的生成图、同域开发集选择和单次种子实验范围会随最终结果一并说明。

本次已通过全部 130 项 CPU 测试，以及 Trainroom 前 40 个真实更新的有限损失、梯度和完整状态检查。
正式任务 `eca6f5624673` 从 step 40 恢复，日志为 `runs/boq_b2_night_only_v1/train/logs/metrics.jsonl`。
截至本次记录，首轮 908 次更新完成并已进入第 2 轮，SVOX night dev R@1 为 **88.7676%**，
MSLS 诊断 R@1 为 **91.4865%**。同一轮的旧 mixed R@1 为 87.8315%，相差 +0.9360 pp；
完整旧 mixed 的最佳开发成绩为 89.5476%，当前新实验尚未超过该完整训练结果。
这些是中途开发指标，最终表格仍待完整训练和固定评测完成。实际进度以 Trainroom 和
[`night_only_v1.json`](../results/night_only_v1.json) 中注明时点的记录为准。
