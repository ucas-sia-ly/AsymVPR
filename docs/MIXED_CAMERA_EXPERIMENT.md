# 混合训练＋夜间退化对照实验

实验为 `boq_b2_mixed_camera_v1`，使用
[独立配置](/home/admin123/github/AsymVPR/configs/boq_b2_mixed_camera_v1.json)。
初始化使用原始 `checkpoints/boq_b2_best.pt`，与混合无退化对照保持同一起点。
本轮不从已经适配后的模型继续，以便与混合无退化实验直接比较。

每批 **32 张生成夜图＋32 张全局白天回放图**，一次学生前向；只有夜图应用相机退化。
白天的抽样、排除验证地点的规则、图像预处理与原实验一致。
夜图按“原尺寸 RGB → 七类随机退化 → ToTensor → Resize 322×322 → ImageNet Normalize”处理。
使用已经复制并固定 SHA 的 WORKSPACE exploratory profile，整体触发概率 0.8；
增益/Gamma、饱和度、高光泛光、运动模糊、暗角、散粒/读出噪声、JPEG 的参数与
[退化参数文件](../configs/night_camera_exploratory_v1.json) 一致。

教师输入与预计算特征不附加退化。夜图正目标继续使用原来的、由未退化日夜教师一致性控制的融合：
`normalize((1-w) T(source_day) + w T(night))`，`w=0.5*clip((cos-.2)/.5,0,1)`。
白天回放正目标为其冻结白天教师描述子。
训练仍为 `0.5 * (night_asym + replay_asym)`，全局负地点中心、对角方差、temperature=.05、gamma=15 不变。

保留原混合选模规则：SVOX night dev R@1 最大，且 MSLS R@1 相对初始化下降不超过 1 pp。
保留日夜融合正目标和 MSLS 门槛选模。
验证和固定最终测试均保留原始图像，不附加人工退化。

预算为同一冻结的 30,579 张生成图（29,050 train / 1,529 合成 holdout），
seed=2026，10 epochs / 9,080 updates，290,500 次训练夜图暴露＋290,500 次白天回放暴露。
FP32、两组学习率、epoch cosine 调度、输入大小及模型架构均与 mixed 无退化一致。
训练 worker=8、prefetch=1，验证 worker=4；退化使用独立的图片/epoch 随机种子，不改变回放抽样或模型 RNG。
生成图保留原始 pending_verification 状态。

已有完整结果，R@1（%）：

| 实验 | SVOX night dev | Tokyo105 night |
|---|---:|---:|
| 原始 GSV 模型 | 80.4992 | 83.8095 |
| mixed 无退化 | 89.5476 | 87.6190 |
| mixed＋退化 | 91.4197 | 84.7619 |

SVOX 是参与选模的同域开发集，Tokyo105 是重复使用的固定最终诊断。
本轮直接对照 mixed 无退化：开发集 R@1 提高 1.8721 pp，Tokyo105 night 降低 2.8571 pp。

启动与恢复都使用同一个配置和新输出目录，由 Trainroom 托管：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR --name 'AsymVPR mixed-camera v1' --total-steps 9080 \
  -- /bin/bash reproduction/mixed_camera_launch.sh \
  --config configs/boq_b2_mixed_camera_v1.json --stage train
```

启动前先查已有任务，避免重复。运行通过输出目录锁、不可变配置、来源哈希保护。
完整状态保存模型、优化器、调度器、计数和 RNG；前 40 步检查后从同一状态继续。
systemd scope 为 MemoryHigh=8 GiB、MemoryMax=10 GiB、MemorySwapMax=0。

关键日志记录真实 global step、总损失及 night/replay 分项、实际夜图退化比例、总图片/秒与夜图/秒、
梯度、两组学习率、显存，以及按实际验证时点写入的 Recall@1/5/10。

完整训练后自动进行同协议五集评测及 Tokyo105 夜间诊断。
`runs/boq_b2_mixed_camera_v1/NIGHT_REPORT.md` 对照原始模型与 mixed best；
`MIXED_CAMERA_REPORT.md` 专门报告 mixed 无退化 → mixed 有退化的差异。
`final_evaluation/mixed_camera_comparison.json` 和 `mixed_camera_finalization.json` 绑定预测、参考与报告哈希。
当前只保留原始基础、混合和混合＋退化；其他方向见“已放弃”。

整理后仅支持混合训练流程。旧实验已完成，不重复启动；如需新实验，使用新的输出目录，保持旧配置、权重、来源哈希和结果收据不变。历史来源快照归档于 `已放弃/cleanup_20261107/retained_run_source_snapshots/`。
