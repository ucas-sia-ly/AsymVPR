# 已放弃：night-only 系列

状态：**已放弃，后续不再启用**。这里仅保存最终结果和选中 checkpoint，不提供训练、恢复或测试入口。
按用户要求清理独立训练代码、配置和原始训练/评估目录；保留混合训练与混合＋退化流程。

| 实验 | 选中 epoch / step | SVOX night dev R@1 | Tokyo105 night R@1 | checkpoint |
|---|---|---:|---:|---|
| boq_b2_night_only_v1 | 6 / 5448 | 89.5476% | 86.6667% | [night_only/checkpoints/best.pt](night_only/checkpoints/best.pt) |
| boq_b2_night_camera_v1 | 10 / 9080 | 90.6396% | 86.6667% | [night_only_camera/checkpoints/best.pt](night_only_camera/checkpoints/best.pt) |

与保留的 mixed 无退化参考相比，SVOX 开发成绩为 89.5476%，Tokyo105 为 87.6190%。退化版虽提高开发集成绩，但未改善该固定夜间测试；此处记录用户的停用决定，不将单种子结果解释为统计显著性。

| R@1 (%) | night-only | night-only＋退化 |
|---|---:|---:|
| msls | 91.7568 | 91.8919 |
| pitts250k | 94.2271 | 94.3478 |
| tokyo247 | 93.0159 | 92.6984 |
| nordland | 78.0797 | 78.8406 |
| amstertime | 51.3404 | 49.2283 |

每个子目录包含：选中 `checkpoints/best.pt`、`checkpoint_summary.json`、最终 `results/summary.json`、`comparison.json` 和五集 top-10 预测。
原始 checkpoint 字节保持不变，内嵌配置仅作历史来源，不代表现存或可运行入口。只保留选中最佳推理权重，删除 optimizer/last 和逐 epoch 权重。
SVOX 是选模开发集，Tokyo105 为固定最终诊断；图库/查询协议与原实验相同。旧结果中的绝对路径是历史信息，原始 run 已退役，不再用于后续实验。

完整 R@1/5/10、teacher/输入身份及 checkpoint SHA256 见 [summary.json](summary.json)。文件校验见 [archive_manifest.json](archive_manifest.json)。

权重的持久归档位于 `/home/admin123/github/AsymVPR/已放弃/night_only/`；工作树中的 checkpoint 为指向该归档的链接，避免工作树清理丢失权重。
