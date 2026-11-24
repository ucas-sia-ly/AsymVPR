# DINOv2-BoQ 的 C 版退化蒸馏对照

用户确认：迁移 C 版自适应退化策略，不加载 EfficientViT 学生权重。
起点为 `runs/boq_dinob_official_real_memory_v2/real/best.pt`；教师和待训练模型
加载相同的本地 8192 维 DINOv2-BoQ 状态，教师全部冻结并始终处于 eval。

这是已训练模型的独立微调实验，不是重新定义原 BoQ 从头训练配方。

- 数据：既有 GSV 真实图及冻结快照 train 划分的生成夜图；真实:生成=8:1。
- 每批：64 地点 × 4 张不同图片，保留尾批；62,514 地点，每轮 977 步。
- C 退化：直接复用 `gsv_night_camera.GSVCamera` 的 adjusted 模式及原冻结阈值
  0.09361752308905125；输入 322×322，与 C 的退化尺度一致；仅夜图分支退化。
- 待训练部分：DINOv2 最后两个 block 和 BoQ 头。AdamW，两组 lr=1e-5、wd=1e-4，
  不再预热，固定学习率，16-mixed。每组 10 轮 / 9,770 步，seed=2024。
- 检索目标：沿用 MultiSimilarityLoss 和 miner，对整个批次共同挖掘正负样本。
- 蒸馏目标：实际发生 C 退化的夜图描述子，对齐冻结教师读取对应干净夜图的描述子，
  `mean(1-cosine(student_degraded, teacher_clean))`。未退化/回退样本不计算蒸馏项。
- 配对对照：control 权重为 0；distill 权重为 1。教师在两组都计算，数据顺序、
  初始化、退化随机种子及训练预算相同；仅损失权重不同。
- 验证：微调前基线及每轮 MSLS-val、Pitts30k-val、SVOX-night-dev（百分数）。
  为避免更改选模标准，仍按 MSLS-val R@1 存 best；夜间效果同时比较同轮结果，
  不只比较各自 best，不运行测试集来选择参数。

训练日志分别记录总 loss、MS loss、原始/加权蒸馏 loss、实际退化夜图数、生成比例、
学习率、吞吐和验证 Recall。baseline.json 是实测微调前成绩，epoch_XX.json 是后续成绩。
微调结果尚不能用于宣称退化蒸馏有效，需等待配对实验完成；单种子也不能证明普遍收益。

配置：`configs/boq_dinob_c_distillation.json`。
启动入口：`python -m reproduction.boq_c_distillation_pair`，由 Trainroom 托管，
先 distill，成功后自动运行 control；失败时停止调度，不重复启动。
结果目录：`runs/boq_dinob_c_distillation_v1/`。


## 后续 11 项：仅无蒸馏，支持恢复

按用户最新要求，后续仅保留 11 项逐项去除的无蒸馏组（distill_weight=0），每组仍为
10 epochs / 9,770 steps。与完整 C 版无蒸馏组按相同轮数比较，不再计算蒸馏交互贡献。
当前完整 C 版任务不因本次修改中断；旧 22 组等待队列已停止，新队列等待完整 C 版结束后串行执行。

目录：`runs/boq_dinob_c_ablation_control_v2/`；Trainroom 队列 ID：`2935419e2735`。
恢复适配器只用于后续 11 项，未修改正在运行的完整 C 版训练源码。

每 100 步另存完整 `resume/last.ckpt`，epoch 末仍保存 `checkpoints/last.ckpt`。
重启时选 global_step 最新的文件，通过 Lightning 恢复模型、优化器、调度器、AMP 与循环进度；
额外保存主进程 Python/NumPy/Torch/CUDA 随机状态，并按采样 epoch 和已完成 batch 跳过输入。
最多重放检查点之后尚未保存的 99 步；多 worker 的随机增强不承诺逐位复现。

后续中断后，在项目目录使用相同 BoQ Python 环境及 Trainroom 托管运行：

```bash
python -m reproduction.boq_c_distillation_ablation --run --resume
```

重连运行中任务，跳过成功完成项；停止/失败项以新 Trainroom 运行记录继续原输出目录，
保留旧任务 ID 历史。配置、权重或源码身份不符时拒绝恢复，不静默覆盖证据。
只需重连仍在运行的队列时优先查看其现有任务，不重复启动队列。

已做离线检查点选择、输入跳过及控制组配置测试；用户未要求现在中断，因此没有实际停止
GPU 训练来验证恢复。结果陆续写入新目录的 `results.json` 与 `REPORT.md`。


## 多夜间协议补评测（训练已暂停）

用户要求暂停消融队列及正在训练的降分辨率组，先评测已完成模型。
暂停时最新训练日志为 step 1960，磁盘完整恢复点为 step 1954（含优化器、调度器和随机状态）。
不自动恢复训练；后续可用原队列的 `--run --resume` 继续。

评测入口 `reproduction.boq_c_night_evaluation`，输出
`runs/boq_c_multi_night_epoch10_v2/`。共六个模型：微调前基线、完整C有/无蒸馏、
去掉降饱和/白平衡/失焦。所有微调模型固定第10轮，不利用新增测试结果选模。
每个模型用自身编码图库与查询；统一 uint8 Resize322、FP32、关闭TF32、精确检索。
不复用之前不同维度教师的图库特征。

覆盖 Tokyo105 night、SVOX test-night、RobotCar-Seasons public Night/Night-rain、
Gardens Point 两种白天图库、MSLS val/test 夜间子集、DarkDriving配对诊断；
SPED单独作为混合光照诊断。RobotCar GPS诊断缺失本地CSV，明确列为不可用。
评测保存 Recall@1/5/10、逐查询命中、top-10、权重及预测和特征哈希，支持中断续提。
Trainroom ID：`32cff9431d1c`。


### 评测中断及恢复

任务 `32cff9431d1c` 已完成 48/60 项，在 DarkDriving 夜图
`Night/in_1_1728606685494302976.jpg` 被 torchvision 判为 incomplete/truncated 时退出。
该原始文件在 `PIL.ImageFile.LOAD_TRUNCATED_IMAGES=False` 下可严格解码。
这是解码器兼容性问题；未修改图片，未删除查询，也未修改对应真值。

SPED 补评测 `e64c4098938c` 已完成 6/6 项；累计已有 54 项有效结果。
DarkDriving 恢复任务 `364d953c6d28` 使用独立缓存和显式严格 Pillow 回退，
仅对 torchvision 的 incomplete/truncated 异常触发，其他异常仍报错。
逐文件 SHA256、路径、解码方式写入恢复目录 decoder_fallback.jsonl。
回退预处理仍为同一 uint8 Resize / Normalize；不启用截断容忍。
已验证兼容分支的 RGB 张量、预处理和证据记录，并以实际问题图片验证严格 Pillow 解码。

完成后汇总入口为 `runs/boq_c_multi_night_epoch10_v2/combined_summary.json` 和
`runs/boq_c_multi_night_epoch10_v2/consolidated/REPORT.md`，绑定各次评测原始汇总哈希。
原来 48 项不重新推理，SPED 与 DarkDriving 从剩余任务接续；训练仍暂停。


### 用户要求继续消融及最终多夜间测试

续训队列 `43471b4ba042` 从降分辨率组的完整 step1954 检查点恢复，新的训练 ID
为 `a244c65425ce`。已完成的前三项直接跳过，剩余无蒸馏消融串行继续。

后续评测任务 `399cb846d19d` 等待整个队列成功后，自动对剩余八个模型的固定第10轮
权重进行同一套10项夜间/混合光照协议评测（80项）。使用已验证的严格 Pillow 回退，
保留 DarkDriving 全部查询。失败时停止自动衔接，不自动改训练或评测协议。
已完成的60项结果复用，最终形成14个模型×10项协议的140项汇总；新增夜间测试结果
不用于重选检查点。
新增评测入口 `reproduction.boq_c_ablation_final_nights --wait-for 43471b4ba042`，
结果目录 `runs/boq_c_remaining_ablations_night_v1/`，完整表在其 `all_models/REPORT.md`。

### 2026-11-22：改用夜间开发集选模，全部关闭蒸馏

用户确认重跑缺失夜间最佳权重的实验，并明确完整 C 也不运行蒸馏。
上节旧续训队列、降分辨率训练及末轮评测等待任务均已停止；历史结果保留。

新队列 `12fb46d19f37` 在 `runs/boq_c_night_selected_control_v1/` 独立运行：
完整 C 无蒸馏 + 11 项逐项去除退化的无蒸馏消融，每组10轮。
只优化原检索损失，不执行蒸馏教师前向。首组完整 C 任务为 `3b880b9c37f0`。
仅以 `svox-night-dev/R@1` 保存最佳权重，MSLS/Pitts30k 仅作观察；
初始模型也参与候选，分数相同优先较早模型。保留完整状态周期检查点供中断恢复。
旧实验的夜间最佳轮次权重已被覆盖，不能以旧 MSLS 最佳或末轮替代。

评测等待任务 `6b7a7897d433` 在新队列全部成功后，自动对初始基线及12组模型
运行同一套10项夜间/混合光照评测，共130项；不包含任何蒸馏模型。
结果独立保存到 `runs/boq_night_selected_multi_night_v1/`。
夜间测试集只报告结果，不参与选模；历史固定第10轮结果与本轮选模结果分开保存。

### 2026-11-23：修复训练结束后的权重导出

完整 C 已完成全部9770步，但导出时原入口保留的回调对象被 NightTrainer 替换，
其 `best_model_path` 为空，导致导出报错，随后队列及评测等待任务退出。
修复为在原回调对象上设置夜间指标，保留入口的对象引用。
新增 CPU 两轮集成检查覆盖保存、最佳轮次选择及结束后读取权重，4项检查通过。

已从保留的夜间最佳完整检查点恢复导出，没有重复训练或新增更新。
完整 C 最佳第3轮，SVOX-night-dev R@1=95.94384%，初始模型为95.16381%。
恢复导出任务 `92a16d23e254`；新训练队列 `4948a0bba638`，
新最终评测等待任务 `667e6752d37f`。全部实验仍关闭蒸馏。
修复前计划及源码哈希变更记录在
`runs/boq_c_night_selected_control_v1/export_callback_repair_20261123.json`，
原配置及已训练权重身份保持不变。

### 2026-11-24：追加匹配的无退化对照

用户授权在当前全部无蒸馏消融结束后，追加无退化对照，再统一多夜间测试。
独立配置 `runs/boq_no_degradation_control_v1/config.json` 复用完整 C 的初始权重、
真实:生成8:1、采样、优化器、10轮9770步和夜间开发集选模；仅将11项操作全部 omit，
关闭额外 C 退化。生成图片仍保留；原基础图像预处理保持一致，蒸馏仍关闭。
已实际检查8张生成图的输出与 clean 张量完全相等，且 changed=False。

入口 `reproduction.boq_no_degradation_followup --wait-for 4948a0bba638` 等待原队列后
启动该对照，并在成功后评测初始基线+12组消融+无退化对照，共14模型×10协议。
原仅等待12组的评测任务 `667e6752d37f` 已停止，以避免与追加对照同时占用GPU。
不更改当前消融队列及其配置和运行状态；训练子任务支持原完整检查点恢复机制。
