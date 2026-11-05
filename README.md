# AsymVPR 论文复现

复现 Kim、Cho、Yoon 的 [Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](https://arxiv.org/abs/2512.13055)。
当前实现和评测的是论文主配置 **DINOv2-B + BoQ 图库 → EfficientViT-B2 + BoQ 查询**。
论文所列作者仓库 `jaeyoon1603/AsymVPR` 在本次核对时返回 404；这里是依据论文公式的独立实现。

本机此前已在 `../Bag-of-Queries` 完成这组 15 epochs 训练。本项目复用已完成的权重、缓存和原始日志，
没有重复启动训练。训练来源与文件 SHA256 见 [training_history.json](results/training_history.json)。
另外三种 SALAD/MobileViTv2 组合和论文全部消融不属于当前已完成实验。

## 生成夜图后续微调

已新增 [夜间适配流程](docs/NIGHT_ADAPTATION.md)，从原 epoch 9 最佳权重继续训练，
固定使用 `WORKSPACE/outputs/night_place_fast_v1` 和 `night_cluster_v1` 中已完成的 **30,579** 张生成夜图。
其中 29,050 张训练、1,529 张按地点留出；配合全局白天回放与冻结教师的日夜融合目标。
真实 SVOX night 开发集用于选模，MSLS val 保留能力约束为相对初始值下降不超过 1 个百分点。

本轮计划 10 epochs / 9,080 updates，FP32，32 night + 32 day。
已通过前 40 个真实更新及完整状态恢复检查，正式任务 `539c22d03c8d` 已从 step 40 续跑。
进度以 [Trainroom](http://127.0.0.1:8765) 为准；结构化日志为
`runs/boq_b2_night_v1/train/logs/metrics.jsonl`，配置为 [boq_b2_night_v1.json](configs/boq_b2_night_v1.json)。
训练完成后自动执行固定最终评测并生成 `runs/boq_b2_night_v1/NIGHT_REPORT.md`。
生成图仍保留原待语义审核状态；教师一致性权重不等同于审核通过。

## 已完成的训练结果

| MSLS validation，Recall % | R@1 | R@5 | R@10 |
| --- | ---: | ---: | ---: |
| 论文 BoQ → EfficientViT-B2 | 92.3 | 95.9 | — |
| 本机最佳权重，epoch 9 | 92.1622 | 96.2162 | 96.3514 |
| 本机最后一轮，epoch 15 | 91.8919 | 96.2162 | 96.6216 |
| 本机冻结教师，对称检索 | 93.7838 | 96.7568 | 97.0270 |

最佳 R@1 与论文相差 −0.138 个百分点。MSLS 同时用于选择最佳 epoch，因此这是验证成绩。
本次重新提取 MSLS 教师图库、重新加载最佳学生权重，独立评测精确重现上述最佳 R@1/5/10。
新结果、输入来源和预测见 [evaluation/summary.json](results/evaluation/summary.json)，报告见 [REPORT.md](results/REPORT.md)。
原 15 epochs 验证曲线见 [msls_training.png](results/msls_training.png)。

本次五集独立评测全部完成，R@1 / R@5（%）为：

| 数据集 | R@1 | R@5 |
| --- | ---: | ---: |
| Pitts250k test | 94.7947 | 98.2971 |
| MSLS val | 92.1622 | 96.2162 |
| Tokyo24/7 | 91.7460 | 96.1905 |
| Nordland，±3 帧 | 78.5145 | 88.9130 |
| AmsterTime | 47.6036 | 69.2933 |

Nordland 对同一预测改用发布的 ±1 帧真值时，R@1 为 72.2101%；主结果采用论文文字中的 ±3 帧。
作者没有明确完整文件列表与季节方向，故不能把本次高于论文的 Nordland 数值解释为方法改进。

- 推理权重：[checkpoints/boq_b2_best.pt](checkpoints/boq_b2_best.pt)，epoch 9、step 74,466。
- 原训练：15 epochs、124,110 次优化器更新；最后完整状态在 `artifacts/previous_training/last.pt`。
- 原训练日志：`artifacts/training_logs/metrics.jsonl`，原任务 ID `223b8a23c9b2`，已成功结束。
- 本次五集评测任务 ID `0be0f8e4e2e9`，状态 completed、exit code 0；耗时约 16.4 分钟，复用了已重提的 MSLS 特征。
- [Trainroom 本机看板](http://127.0.0.1:8765)：查看训练 loss、梯度范数、学习率、吞吐和 MSLS Recall。

## 运行环境

本机已验证的解释器：

```bash
/home/admin123/miniconda3/envs/boq/bin/python -m reproduction.preflight
/home/admin123/miniconda3/envs/boq/bin/python -m pytest -q tests
```

环境为 Python 3.12、PyTorch 2.5.1+cu124、torchvision 0.20.1、timm 1.0.22。
默认 base Python 使用 cu130，与当前驱动不兼容；直接用上述已有 `boq` 环境。
历史环境完整清单保存在 `artifacts/previous_training/requirements-lock.txt`。
新机器安装示例：

```bash
python3.12 -m venv .venv
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/pip install -r requirements.txt
```

`configs/boq_b2.json` 使用本机已有数据、BoQ 教师权重和 DINOv2 源码路径；迁移时调整这些路径。
输入为 RGB、322×322，tensor bicubic resize、antialias、ImageNet normalization。

## 方法与训练

冻结图库模型，离线提取全部训练图像的 12,288 维 L2 归一化描述子。
按 GSV-Cities 的 city/place ID 保存算术均值与对角样本协方差，中心不再归一化。
查询模型使用 ImageNet 预训练 EfficientViT-B2 和同类型 BoQ 聚合器，共 22,275,984 个参数。

对查询 `q`、同一图像的教师特征 `g`、地点 `y`，Eq. (8) 为：

```text
positive = dot(q, g) / tau
negative = logsumexp(dot(q, c_j) / tau, all j != y)
augmentation = gamma / (2 * tau^2) * sum(q^2 * variance_y)
loss = mean(softplus(negative - positive + augmentation))
```

使用全部地点负样本，分块精确合并 logsumexp；不做 kNN 挖掘或队列替换。
`tau=0.05`、`gamma=15`；`gamma=0` 对应去掉隐式增强的消融。
AdamW、batch 64、15 epochs，学习率从 5e-4 逐 epoch cosine 衰减至 1e-4。
FP32，关闭 TF32；不自动使用 AMP，不改变 batch 内负样本语义。

当前训练已完成。明确需要新实验时才执行以下命令；默认新输出目录与历史训练分离：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py ensure
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR --name 'AsymVPR BoQ-B2 new training' --total-steps 124110 \
  -- /bin/bash reproduction/launch.sh --config configs/boq_b2.json
```

准备、训练、恢复由 `python -m reproduction.run --stage {manifest,prepare,train,all}` 控制。
完整 checkpoint 保存模型、优化器、scheduler、随机状态和下一 batch；配置和缓存来源不符时拒绝恢复。
已存在完成标记时不重复训练。新实验应使用新的 output/cache 目录。
训练日志可脱离本机 SDK 写兼容 JSONL；通过 Trainroom 托管时复用其原生记录器。

## 测试集评测

`python -m reproduction.evaluate --help` 查看完整参数。
评测固定使用教师编码图库、最佳学生编码查询，报告 Recall@1/5/10。
分批提取和精确检索使用 mmap/分块；缓存校验模型哈希、输入处理及有序图像列表，支持中断续提。
测试集结果只作报告，不用于选择 checkpoint。

本机通过 Trainroom 跑全五集并生成报告：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR --name 'AsymVPR epoch9 five benchmark suite' \
  -- /usr/bin/env ASYMVPR_EVAL_MODULE=reproduction.suite /bin/bash reproduction/eval_launch.sh \
  --config configs/boq_b2.json --checkpoint checkpoints/boq_b2_best.pt \
  --dataset all --cache-dir results/evaluation_cache --output-dir results/evaluation \
  --batch-size 128 --workers 4
```

相同参数会复用来源和校验和一致的已完成特征，不重新训练。执行前先确认同一评测任务已退出。
仅检查五集文件和协议可加 `--manifest-only`，不会加载模型。

| 数据集 | 正匹配规则 | 本机数据 |
| --- | --- | --- |
| MSLS val | 官方 25m positives，18,871 refs / 740 queries | `../datasets/msls_val` |
| Pitts250k test | 官方 25m GT，83,952 refs / 8,280 queries | `../datasets/Pittsburgh/Pittsburgh` |
| Tokyo24/7 | 标准 315 queries，UTM 25m | `../datasets/tokyo247` |
| Nordland | 论文 ±3 帧；具体发布与季节方向写入结果 | 本机 VPR-Bench 原始 release |
| AmsterTime | 官方 image_pairs 对应关系 | `../datasets/amstertime` |

Nordland 不能把不同季节方向、JPEG 重编码和 ±1/±3 帧容差的结果混为同一协议。
论文没有完全公开其文件清单和季节方向；评测结果记录本次实际使用的协议。

## 复现边界与资源

论文写约 560k 图像、67k 地点；本机完整可用 GSV 元数据为 **529,506 图像、64,394 地点、23 城市**。
所有图像都参与训练，没有为跑通而缩减样本。此数据差异限制与论文的严格数值一致性。
其他未公开细节的实现选择是：weight decay 0.01、seed 2024、逐图像 shuffle、无额外随机增强、
协方差以 n−1 估计、BoQ 聚合器随机初始化。cuDNN benchmark 开启，不能承诺位级重现。

本机实际检测为 RTX 4090，约 48 GiB 显存、30 GiB 系统 RAM。
历史正式训练峰值 PyTorch 已分配显存约 25.2 GiB，训练末期约 223 images/s；这不是端侧推理延迟。
本次 batch=1 网络前向中位延迟为教师 **9.061 ms**、查询 **3.502 ms**，同机耗时比 **2.59×**。
计时为 10 次预热、50 次 CUDA 同步 wall clock，重复合成 GPU 输入，FP32；不含读图、预处理、传输和检索。
这是本机桌面 GPU 计时，不能直接当作论文硬件上的延迟，完整样本见 [efficiency.json](results/efficiency.json)。
训练特征约 24.2 GiB、地点均值及方差合计约 5.9 GiB，保存在磁盘 mmap，避免全量驻留 RAM。
历史特征、训练状态和日志使用 `artifacts/` 软链接，不会额外复制几十 GB 数据；最佳推理权重单独复制。
移动或删除 `../Bag-of-Queries` 前，应先保存这些链接所指的文件。

`src/` 保留 BoQ 官方实现和许可，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
