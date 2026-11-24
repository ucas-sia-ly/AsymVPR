# 项目维护与参考实现

核对日期：2026-11-21。已有实验保持封存，以下修改不代表重新训练或取得新的 Recall。

## BoQ 训练与生成图比例

新的标准训练入口是 `reproduction.train_dino_gsv_mixed`，默认配置为
[boq_dinob_official_ratios.json](../configs/boq_dinob_official_ratios.json)。
对照 [BoQ 官方 train.py](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/train.py)、
[model.py](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/src/model.py) 和
[datamodule.py](https://github.com/amaralibey/Bag-of-Queries/blob/1a4965ea7dfd9bd0dd846adf7a0e430f68101d12/src/dataloaders/datamodule.py)，
优化器、预热、loss/miner 和增强直接沿用官方实现；本地适配数据路径、日志、实验身份和生成图采样。

| 项目 | 当前默认值（与固定版本官方训练入口一致） |
| --- | --- |
| 初始化 | DINOv2-B 原始预训练权重，随机 BoQ 头，seed 2024 |
| 可训练部分 | DINOv2 最后 2 个 block，以及 BoQ 聚合器 |
| 聚合器 | 512 通道、64 queries、2 layers、8192 维输出 |
| 批次 | 128 个地点 × 每地点 4 张图 = 512 张；最后不足 128 地点的批次保留 |
| 地点范围 | 所有至少有 4 张真实图的 GSV 地点；不局限于有生成图的地点 |
| 地点顺序 | 每个城市先打乱真实图片行，再取地点首次出现顺序；每轮刷新，保持官方的城市内采样方式 |
| 尺寸 | 训练 224×224，验证 322×322 |
| 训练增强 | RGB uint8 tensor → bicubic Resize → RandAugment(3,15,bilinear) → float32 → ImageNet Normalize |
| 优化器 | AdamW；骨干与聚合器均为 lr=1e-4，weight decay=1e-4 |
| 日程 | 40 epochs；前 10 epochs 逐步线性预热；MultiStepLR 在 10/20 epoch 乘 0.1 |
| 损失与挖掘 | MultiSimilarityLoss(alpha=1,beta=50,base=0)，Miner(epsilon=0.1) |
| 精度 | Lightning 2.4.0 的 16-mixed；不额外裁剪梯度，不拆分批内负样本，不增加梯度累积 |
| 验证与选模 | 每 epoch 验证 MSLS-val 和 Pitts30k-val；MSLS-val R@1 选 top 3，完整 last.ckpt 可恢复 |

这是官方源码默认 8192 维训练配方；已有 AsymVPR 复现使用的冻结 12288 维教师、
EfficientViT 学生及其蒸馏实验不因此改成另一套架构或训练目标。
AdaptVPR 论文所写的 BoQ 训练尺寸是 280×280，而上述固定版本官方源码使用 224×224；
这里按用户要求优先沿用 BoQ 官方源码，不把两者说成完全相同。

参考 [AdaptVPR §III-F、§IV-D 表 IX](https://arxiv.org/html/2609.04369v1#S4.T9)，
固定生成图池、地点范围、初始化和优化步数，仅比较 **纯真实、真实:生成=8:1、4:1**。
论文的 8:1 最优结论只作为参考，未在本项目重新验证，不预先认定某个比例更好。
本项目按用户选择，每地点维持 4 张且不重复；当前每地点最多用 1 张生成图，
其余为不同真实视角，不对同一生成文件重复采样，2:1 和 1:1 不纳入本轮。
生成图读取既有冻结快照的 train 划分，保留上游审核字段，不新增审核筛选。

512 不能被 9 或 5 整除，完整批次采用 56/57 张或 102/103 张生成图的累计整数配额。
缺少生成图的地点和尾批仍保留，必要时将不足配额分配到有余量的批次；
记录每 epoch 的实际真实/生成计数，整体比例误差少于 1 张图。容量不足时明确报错。
本地只做 CPU 元数据预检，三组均为 **62,514 地点、489 steps/epoch、19,560 总 steps**，
每 epoch 250,056 张图；8:1 使用 27,784 张生成图，4:1 使用 50,011 张。

仅预检，不加载模型或启动训练：

```bash
python -m reproduction.dino_gsv_ratio_sweep --preflight
```

后续只有获得训练授权才通过 Trainroom 托管启动。训练依赖见
[requirements-training.txt](../requirements-training.txt)。
检查点按官方 Lightning 流程在完整 epoch 后保存；恢复优化器、调度器、AMP 状态和采样 epoch，
不承诺随机增强及 CUDA 运算逐位重现。
真实日志含 loss、学习率、实际生成比例、每批真实/生成图数、挖出的正例对数、吞吐和
各验证集的 R@1/5/10（百分数）；不把合成标量当训练结果。

旧的 `boq_dinob_gsv_ratio_sweep.json` / `boq_dinob_gsv_ratio_full10.json` 仍保留历史记录用途，
它们曾使用 90 张图、每地点 2 张、短预热、不同学习率且没有官方增强，不能充当官方 BoQ 基线；
新入口拒绝按这些旧配置启动或续训。新输出固定为 `runs/boq_dinob_official_ratios_v1/`。

## 模型代码整理

- `BoQ.forward(..., return_attention=False)` 保留可选的描述子推理接口；标准 BoQ 训练和验证
  都使用官方默认注意力路径。已有非对称模型的 eval 可省略未使用的注意力图，未测 GPU 加速比。
- 删除无调用方的 ResNet 包装器和一次性的 `benchmark_gallery.py`；正式计时入口仍为
  `reproduction.efficiency`。
- `DinoV2(dino=...)` 接收已加载骨干；修复解冻 0 个 block 时 `[-0:]` 意外选择全部层的问题。

## 评测与清理

`reproduction.night_suite` 默认可独立提取教师图库和各学生查询，不再要求已放弃实验存在。
`--reuse-from` 可显式指定历史评测目录；来源缺缓存时提取，来源冲突时报错。
`--manifest-only` 只检查协议，不加载模型。

本轮删除的仅为“已放弃”下可重建的 `cache`、`final_cache`、`confusion_cache`，
以及 Python 测试缓存和 `tmp/pdfs/method_review.png` 临时预览。
现用 `results/evaluation_cache` 被多个保留实验通过符号链接共享，予以保留；
现用夜间评测目录通过硬链接共享的特征也保留，因此删除旧目录的文件大小总和
不等于实际回收磁盘量。清理统计见 [cleanup_20261121.json](cleanup_20261121.json)。

保留全部权重、训练日志、配置、结果指标、top-k 预测、命中标记和来源快照。
`results/**/evaluation_source` 是历史评测绑定哈希的源码证据，不属于可以直接删除的重复代码。
已有 GSV 对比、相机消融及历史配置保留；比例训练入口按本次要求改为官方 BoQ 配方。

## 验证

CPU 测试共 208 项通过，覆盖官方增强、优化器和学习率日程、采样比例、四视图唯一性、
完整地点范围、尾批、断点身份及已有非对称流程；14 项默认参数与固定版本官方入口逐项相符。
另核对 MSLS-val（18,871 图库 / 740 查询）和 Pitts30k-val（10,000 / 7,608）的本地路径与真值，
从本地预训练权重构建了模型，确认输出 8192 维、只解冻第 10/11 个 DINOv2 block（从 0 开始）。
上述检查未执行模型前向，也未调用 Trainer.fit。
未启动训练、完整数据集重新评测或 GPU 性能试跑，未声称新的 Recall 或速度收益。

## 32GB 主机内存修复

首个纯真实训练在 step 489 后进入验证时发生主机 OOM，systemd 记录 `oom-kill`，未保存首轮检查点。
修复采用 2 个训练 worker、prefetch=1、关闭 pin_memory；验证使用主进程读取。
验证按图片顺序逐批建立精确 FAISS L2 图库并搜索查询，仅保留 top-10 预测，
不积累所有描述子、不执行整套 `torch.cat`，每个数据集完成后立即释放图库。
训练仍为 128 地点×4 张图，官方增强、损失、精度和学习率日程不变。
修复版配置为 `configs/boq_dinob_official_real_memory_v2.json`，独立输出目录；
Trainroom 任务 `c1b02401d761`，scope 限制 MemoryHigh=7GiB、MemoryMax=9GiB、MemorySwapMax=0。
流式验证与完整 FAISS 检索的回归测试一致。旧任务 `39ccc917dca8` 保留异常日志。

修复版实跑已完成首轮训练、MSLS/Pitts30k 验证并进入第 2 轮（step 490）；
整个进程组观测峰值约 7.00 GiB，训练/验证无新增 OOM。首轮 MSLS-val R@1 为 70.27%，
Pitts30k-val R@1 为 90.79%；属于早期验证结果，不代表最终收敛精度。全量测试 209 项通过。
