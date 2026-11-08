# BoQ → EfficientViT-B2 生成夜图微调

本轮在原 AsymVPR 独立复现的最佳查询模型上继续训练，保留冻结的 DINOv2-B + BoQ 图库模型及其 12,288 维描述子空间。初始化文件为 `checkpoints/boq_b2_best.pt`，来源是原 15 epochs 实验的 epoch 9、step 74,466。新的 AdamW 优化器、学习率调度器和随机状态属于独立实验 `boq_b2_night_v1`；完整参数见 [`configs/boq_b2_night_v1.json`](../configs/boq_b2_night_v1.json)。

微调入口为 [`reproduction/night_train.py`](../reproduction/night_train.py)，数据冻结、损失和开发集评测分别实现于 `night_data.py`、`night_objective.py`、`night_eval.py`。此实验增加生成夜图与白天回放，属于原方法基础上的后续适配实验。

本次正式 Trainroom 任务为 **`539c22d03c8d`**，在 [本机看板](http://127.0.0.1:8765) 查看实时状态。
前 40 次真实更新已通过有限梯度、资源及完整断点检查，正式任务从第 40 步继续，没有重复训练这些 batch。
[完整标量日志](../runs/boq_b2_night_v1/train/logs/metrics.jsonl) 保留原 step 0/1/20/40 记录后继续追加；
[stdout](/home/admin123/一些设置/终端/train-terminal/.terminal/runs/539c22d03c8d/output.log) 保留实际进程输出。
训练源代码快照与哈希位于 `runs/boq_b2_night_v1/source_snapshot/` 和 `train/lineage.json`，恢复时拒绝混用已改变的训练代码。
启动核验与任务索引见 [night_adaptation_v1.json](../results/night_adaptation_v1.json)。

生成图片来自下面两个真实目录；它们分别保存本机生成结果和第二台设备经校验回收后的结果：

```text
/home/admin123/github/WORKSPACE/outputs/night_place_fast_v1
/home/admin123/github/WORKSPACE/outputs/night_cluster_v1
```

快照读取两边已发布的 `results.jsonl`，保留 `sample_id`、city/place ID、源图与输出图路径、SHA256、生成参数及原审核状态。`incoming/` 传输副本和旧的手动导出 `generated_manifest.jsonl` 不参与采样。每张夜图通过精确源图路径映射到 `artifacts/gallery_cache/paths.npy` 的原始行，并核对 `labels.npy` 和 `place_keys.json`，因此正样本目标与地点编号沿用原全局图库银行。

本轮已封存的输入位于 [`runs/boq_b2_night_v1/inputs`](../runs/boq_b2_night_v1/inputs/summary.json)：

| 项目 | 实际值 |
| --- | ---: |
| 可用生成夜图 / 地点 | 30,579 / 30,579 |
| 微调训练夜图 / 地点 | 29,050 / 29,050 |
| 合成开发夜图 / 地点 | 1,529 / 1,529 |
| 排除样本 | 0 |
| 完整源图和夜图 SHA256、解码及尺寸审计 | 48.16 秒，4 个 CPU worker |
| 冻结元数据文件 | 730 个，约 418 MB |

`records.jsonl` 的 SHA256 为 `5bad1c9e167aba58c0ab177f494902785cb30f1a999c13743e327cfcf63f31cb`。原图和生成图片保留在原位置，快照复制元数据，不复制大图片。所有样本仍保留上游 `status=pending_verification`、`passed=false`、`eligible_for_training=false` 等字段；用户本次授权将这些图用于训练，代码没有将其改写为审核通过。完整哈希、可解码、尺寸和标签一致性能够验证数据身份，生成夜景的真实性与永久结构是否保持尚未经过 LightGlue、CLIP 或 VLM 的完整语义审核。教师一致性权重也不是语义审核结论。

按完整 `(city, place_id)` 对 `SHA256([2026, [city, place_id]])` 排序，前四舍五入的 5% 地点进入合成开发集，其余 95% 进入本轮训练集。相同地点和源图不会跨越本轮划分；白天回放也排除全部合成开发地点。初始化模型此前已经见过这些地点的 GSV 白天图，所以此划分衡量新生成夜间外观的适配效果，不构成未见地点测试。

封存后继续生成的新图片不会自动进入本实验。再次调用数据准备会校验并复用当前快照。输入配置、源缓存身份、静态生成来源、冻结清单或图片发生变化时拒绝混用；需要加入后续新图时，创建新的实验配置、`input_dir`、`output_dir`、夜图教师缓存和评测输出目录。

训练时，冻结教师分别编码夜图与其精确源白天图，记归一化描述子为 `t_night` 和 `t_day`。教师特征预先缓存，不通过训练梯度更新。夜图正目标为：

```text
c = dot(t_day, t_night)
w = 0.5 * clip((c - 0.2) / 0.5, 0, 1)
target = normalize((1 - w) * t_day + w * t_night)
```

教师余弦一致性不超过 0.2 时使用源白天目标；达到 0.7 后夜间目标权重为 0.5。归一化之前的源白天系数始终至少为 0.5，低一致性的夜图仍参与训练。损失为夜图分支和白天回放分支 Eq. (8) 损失的等权平均。两分支都使用原始 64,394 个地点的全部负地点中心，保留原白天均值、对角样本协方差、`tau=0.05`、`gamma=15` 和 4,096 地点分块；没有用夜图重建全局地点银行。

| 训练设置 | 本轮配置 |
| --- | --- |
| epochs / optimizer updates | 10 / 9,080；每 epoch 908 updates |
| 每 epoch 夜图 | 29,050 张训练夜图，随机顺序完整遍历一次 |
| 每 epoch 白天回放 | 从排除开发地点后的全局 GSV 图片均匀、有放回抽取 29,050 张 |
| 一个正常 batch | 32 张夜图 + 32 张白天回放；尾 batch 为 26 + 26 |
| optimizer | AdamW，weight decay 0.01 |
| backbone / aggregator 初始 LR | 1e-5 / 5e-5 |
| scheduler | 每 epoch cosine；10 轮结束时的调度终点为 2e-6 / 1e-5，即各初始 LR 的 0.2 |
| precision | FP32；AMP、matmul TF32、cuDNN TF32 关闭 |
| 输入 | RGB，322×322 tensor bicubic resize、antialias、ImageNet normalize |
| seed / workers / prefetch | 2026 / 4 / 1 |
| CPU threads | 6 |

配置中的实际优化器读取 `lr_backbone`、`lr_aggregator` 和 `min_lr_ratio`；继承的 `lr`、`min_lr` 字段不作为本轮学习率输入。每 epoch 共处理 58,100 张图像视图，回放允许重复抽到某张白天图。cuDNN benchmark 开启，固定种子不能保证跨硬件或软件版本的逐位一致性。

每次训练前先评测初始化模型并保存 `baseline.json`，之后每 epoch 结束评测以下三个集合，均使用冻结教师图库和当前学生查询描述子，报告 Recall@1/5/10（百分数）：

| 集合 | 查询与图库 | 用途 |
| --- | --- | --- |
| synthetic_night | 1,529 张合成开发夜图；30,579 张冻结来源的白天教师描述子作图库与干扰项，同地点为正例 | 监控合成外观适配 |
| msls_guard | 官方 MSLS val，740 queries / 18,871 references，25 m 真值 | 约束原检索能力下降 |
| svox_night_dev | SVOX **train split** 的 641 张真实 night queries / 22,232 张完整 train gallery，25 m UTM 正例 | 真实夜间开发选模 |

SVOX 使用已有冻结 `svox_train_weather.json` 中的 night 子集及其 `protocol.json`，逐项重验文件、真值和地理隔离规则：删除距离任意 SVOX val/test 图库或查询图小于 50 m 的开发查询。图库保留完整 train gallery；这个 50 m 规则施加于开发查询，不声称图库的每张图片都与 val/test 地理隔离。它是 Oxford 同城市、同相机的目标域开发数据，不能据此宣称跨城市泛化或 SVOX test 成绩。其他 weather 域查询不参与本轮选模。

`best.pt` 首先保留初始化模型。只有当前 MSLS R@1 相对初始化基线下降不超过 **1.0 个百分点**，且 SVOX night dev R@1 严格超过当前最佳值，才更新 `best.pt`。合成开发集指标用于诊断。Pitts250k、Tokyo24/7、Nordland、AmsterTime 等测试成绩用于选模完成后的报告，不反馈到本轮最佳权重选择。如果没有满足条件的改进，最佳文件会继续保存初始化权重，并在状态中记录 `adapted=false`。

Trainroom 使用真实 optimizer step 作为训练横轴。准备阶段的特征数量记录在 step 0；训练每 20 updates 记录 loss、night/replay 分项、教师 day/night 余弦、夜间教师平均权重、查询与两种教师的余弦、梯度范数、两组学习率、吞吐及峰值显存。验证指标分别命名为 `val/synthetic_night/R@K`、`val/msls_guard/R@K`、`val/svox_night_dev/R@K`。JSON、top-10 预测和固定教师基线保存在 `runs/boq_b2_night_v1/train/evaluation/`。

所有执行都通过本机 Trainroom 托管。运行相同输出目录的任务前，确认此前的对应进程已经退出；输出锁也会拒绝并行写入同一实验。下面是完整流程入口，已封存的数据与完成的特征缓存会自动复用：

```bash
cd /home/admin123/github/AsymVPR
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py ensure
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR \
  --name 'AsymVPR BoQ-B2 night adaptation v1' --total-steps 9080 \
  -- /usr/bin/env TRAIN_TERMINAL_RUN_DIR=/home/admin123/github/AsymVPR/runs/boq_b2_night_v1/train/logs \
  /bin/bash reproduction/night_launch.sh \
  --config configs/boq_b2_night_v1.json --stage all
```

准备入口为 `--stage prepare`，负责验证冻结快照、提取夜图教师特征并准备 SVOX/MSLS 评测缓存。`--stage snapshot` 只执行 CPU 数据快照。准备结束后，启动或恢复优化可使用：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR \
  --name 'AsymVPR BoQ-B2 night adaptation v1 resume' --total-steps 9080 \
  -- /usr/bin/env TRAIN_TERMINAL_RUN_DIR=/home/admin123/github/AsymVPR/runs/boq_b2_night_v1/train/logs \
  /bin/bash reproduction/night_launch.sh \
  --config configs/boq_b2_night_v1.json --stage train
```

`night_launch.sh` 使用已有 `boq` 环境，systemd scope 设置 MemoryHigh=8 GiB、MemoryMax=10 GiB、禁用该 scope 的 swap；CUDA 分配预算按启动时空闲显存计算并预留约 4 GiB。大特征数组使用磁盘 mmap，固定地点均值与方差放在 GPU。资源预算不构成对其他 GPU 或内存任务的抢占。

稳定日志路径通过上述环境变量传给训练脚本。恢复产生新的受管任务 ID 后，用
`trainroom.py link --id 新任务ID --path /home/admin123/github/AsymVPR/runs/boq_b2_night_v1/train/logs`
将它关联到同一份连续历史；托管 stdout 和停止控制仍属于新任务。不要在旧进程仍运行时重复启动。

`runs/boq_b2_night_v1/train/last.pt` 每 200 updates 和 epoch 边界原子保存模型、optimizer、scheduler、epoch、下一 batch、step、Python/NumPy/PyTorch/CUDA RNG、基线和最佳状态。正常 SIGINT/SIGTERM 请求会在 batch 边界保存再退出；新进程验证完整配置、初始化 checkpoint、输入快照及教师缓存 lineage 后恢复。验证被中断时保留已完成训练 batch 的状态，恢复后重做该 epoch 验证，再提交 scheduler 与 epoch。每 epoch 另存 `epoch_XX.pt` 和对应指标；10 epochs 全部结束才生成 `completed.json`，再次运行已完成实验不会重复训练。

数据审计与教师特征提取完成只表示训练输入准备就绪。优化进度、实际 Recall 改善和最终最佳 epoch 以 Trainroom 指标及本轮 `baseline.json`、`epoch_XX_metrics.json`、`completed.json` 为准。

10 轮完成后，`night_finalize.py` 自动使用选中的 `best.pt` 做固定最终评测。五集教师图库经过来源与完整 SHA 校验后以软链接复用，学生查询特征只写入独立的 `final_cache/`，不覆盖原复现结果。Tokyo24/7 完整 315 张仍按混合时段报告，另依据独立来源核对的 CSV 时间标签和官方 MAT 真值，派生其中 105 张真实夜图的最终诊断；它不参与选模。
最终生成 `runs/boq_b2_night_v1/NIGHT_REPORT.md`、`final_evaluation/comparison.json` 和绑定权重、协议及输出哈希的 `finalization.json`。最终评测中断时，重新运行已完成训练的入口会继续该评测，不重复训练。`--skip-final-evaluation` 只跳过这一步，不更改训练设置。
