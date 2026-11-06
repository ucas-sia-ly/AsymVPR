# 仅 GSV-Cities 训练模型的夜间基线对照

已核对原始 checkpoint 的实际 SHA256 与既有评测、两次微调的初始化 lineage 一致。下表复用已完成并封存的真实评测，没有重复启动训练或 GPU 推理。

原始权重：`/home/admin123/github/AsymVPR/checkpoints/boq_b2_best.pt`（原始 epoch 9）。SHA256：`d051f6cec174614a7fed1391624420308f31cac6aa59782530f582a01eacb54a`。

混合微调选中 epoch 7；night-only 选中 epoch 6。night-only 已完成 10 epochs / 9,080 steps，Trainroom 任务 `eca6f5624673` 正常退出（exit code 0），最终评测封印已核对。

| 夜间评测，R@1 (%) | 仅 GSV-Cities | 混合微调 | 提升 (pp) | night-only | 提升 (pp) |
|---|---:|---:|---:|---:|---:|
| SVOX night dev（641 queries） | 80.4992 | 89.5476 | +9.0484 | 89.5476 | +9.0484 |
| Tokyo24/7 night（105 queries） | 83.8095 | 87.6190 | +3.8095 | 86.6667 | +2.8571 |

| 评测 | 模型 | R@1 (%) | R@5 (%) | R@10 (%) |
|---|---|---:|---:|---:|
| SVOX night dev（641 queries） | 仅 GSV-Cities | 80.4992 | 89.7036 | 92.1997 |
| SVOX night dev（641 queries） | 混合微调 | 89.5476 | 94.6958 | 95.6318 |
| SVOX night dev（641 queries） | night-only | 89.5476 | 94.2278 | 95.6318 |
| Tokyo24/7 night（105 queries） | 仅 GSV-Cities | 83.8095 | 92.3810 | 93.3333 |
| Tokyo24/7 night（105 queries） | 混合微调 | 87.6190 | 94.2857 | 94.2857 |
| Tokyo24/7 night（105 queries） | night-only | 86.6667 | 93.3333 | 94.2857 |

SVOX night R@1 正确检索数：原始 516/641，混合 574/641，night-only 574/641。Tokyo night R@1：原始 88/105，混合 92/105，night-only 91/105。

两种适配都高于仅白天训练的基线；night-only 未超过混合微调：SVOX R@1 持平，Tokyo 夜间少命中 1 张（−0.9524 pp）。这是单次种子结果，不据此声称统计显著性。

SVOX night 是同城、同相机的开发集，参与选择最佳 epoch；Tokyo105 是固定夜间最终诊断，未参与选模。完整 Tokyo315 含不同时段，不作为纯夜间成绩。所有对比保持同一冻结教师图库、查询及真值协议，FP32、关闭 TF32、精确检索。

来源：

- 原始五集评测：[summary.json](/home/admin123/github/AsymVPR/results/evaluation/summary.json)。
- 混合微调：[NIGHT_REPORT.md](/home/admin123/github/AsymVPR/runs/boq_b2_night_v1/NIGHT_REPORT.md)。
- night-only：[NIGHT_REPORT.md](/home/admin123/github/AsymVPR/runs/boq_b2_night_only_v1/NIGHT_REPORT.md)。
- 两次最终比较的封印、实际 best checkpoint SHA、原始初始化 SHA 与 Trainroom 完成状态均已核对。
