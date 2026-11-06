# 夜间相机退化训练实验

实验 `boq_b2_night_camera_v1` 在 night-only 训练基础上加入 WORKSPACE 的七类相机退化。
配置：[boq_b2_night_camera_v1.json](/home/admin123/github/AsymVPR/configs/boq_b2_night_camera_v1.json)。

处理顺序严格为：**原尺寸 RGB 夜图 → 相机退化 → ToTensor → Resize（322×322，bicubic，antialias）→ ImageNet Normalize**。
没有在退化前偷偷缩小图像。模糊长度、泛光半径等参数以原图像素为单位。

实现及参数逐字复制自 WORKSPACE，分别保存到本项目的 `reproduction/camera_augmentation.py`
和 `configs/night_camera_exploratory_v1.json`，避免外部项目后续修改影响本次实验。
实现 SHA256：`59f8360ebe21e3c6831424e8b858b893095b6557e3439281478156a41d0e05d6`；
profile SHA256：`4e9c9b098d46fedd488d17dc9d6ce8e047dfc1504d79318c72c03d41c023f0d5`。

每张图每个 epoch 以 80% 概率进入退化流程，其余保留原图；进入流程后各操作独立抽样：

| 操作 | 流程内概率 | 参数范围 |
|---|---:|---|
| 增益 / Gamma | 100% | gain 1.0–1.4，gamma 0.8–1.0 |
| 高光泛光 Bloom | 50% | threshold 0.8–0.95，strength 0–0.2，radius 1–3 px |
| 运动模糊 | 50% | length 1–5 px，angle −30°–30° |
| 散粒噪声 + 读出噪声 | 80% | shot_peak 200–1000，read_std 0–0.006 |
| JPEG 压缩 | 50% | quality 75–100 |
| 降饱和度 | 100% | factor 0.6–1.0 |
| 暗角 | 50% | strength 0–0.2 |

实际操作顺序沿用 WORKSPACE：gain/gamma → saturation → bloom → motion_blur → vignette → noise → JPEG。
这些参数是 exploratory 模拟，未通过真实相机标定，也未用测试图拟合。亮度增益不能恢复已经截黑的结构。

学生只接收训练夜图；教师目标仍是对应未额外退化夜图的冻结 `T(night)` 特征。
这是让学生在扰动下保持地点描述的训练。教师缓存、图库和真实验证/最终测试查询均不附加这些人工退化。
验证和测试仍保留各自原始图像中的天然噪声、照明和模糊。

与无退化 night-only 对照保持同一原始 epoch 9 初始化、冻结 30,579 张图和地点划分
（29,050 train / 1,529 synthetic holdout）、seed=2026、night batch=32、10 epochs / 9,080 updates、
290,500 次夜图暴露、FP32、学习率和调度、Eq.8 全局银行及纯夜间教师目标。
无白天回放，无 MSLS 选模门槛，最大 SVOX night dev R@1 选择最佳 epoch。
生成图原始 `pending_verification` 状态继续保留。

数据顺序和每张图的退化分开抽样：退化只使用由训练 seed、epoch、不可变夜图 SHA 和源图索引
派生的局部 NumPy 随机数，不消耗模型 Dropout、Python 或 NumPy 全局 RNG。
不同 worker 数与中断后续训保持同一图片的退化结果。

256 张图的 CPU 加载实测（含 worker 启动、原图退化、预处理、IPC 和校验）为：
4 workers 58.22 images/s；8 workers 79.78 images/s。两者张量 SHA 完全一致，均有 205/256 张进入退化。
采用训练 worker=8、prefetch=1；原始验证保持 worker=4。
数据加载速度不等同于完整训练速度；实际训练吞吐和退化比例由 Trainroom 实测。

执行入口：

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR --name 'AsymVPR night-camera v1' --total-steps 9080 \
  -- /bin/bash reproduction/night_camera_launch.sh \
  --config configs/boq_b2_night_camera_v1.json --stage train
```

已有同一输出的任务时应查看该任务；同目录由文件锁和配置/来源哈希保护。
完整恢复保存模型、优化器、调度器、步数和 RNG。运行代码和 profile 也复制进独立 `source_snapshot`。
systemd scope 设 MemoryHigh=8 GiB、MemoryMax=10 GiB、MemorySwapMax=0。

关键曲线包括 night loss、SVOX night dev R@1/5/10、合成 holdout Recall、MSLS 诊断、两组学习率、
梯度范数、实际 images/s、显存，以及每个记录窗口真实的 `train/camera_applied_fraction`。
验证指标按实际 optimizer step 写入，不把初始化指标重复当作当前值。

完整训练后自动执行同协议五集和固定 Tokyo105 夜间最终诊断：

- `runs/boq_b2_night_camera_v1/NIGHT_REPORT.md`：原始模型和 mixed best 对照。
- `runs/boq_b2_night_camera_v1/CAMERA_REPORT.md`：无退化 night-only best 与本次最佳权重的直接对照。
- `final_evaluation/camera_comparison.json` 与 `camera_finalization.json`：含参考权重、协议、预测和结果哈希的封存记录。

无退化 night-only 的完整对照选中 epoch 6，SVOX night dev R@1=89.5476%，Tokyo105 R@1=86.6667%。
该参考的实际 best、完成标记、已封存结果及预测都经过检查，并由本次配置中的 SHA 固定。
SVOX 属于选模开发集；Tokyo105 是重复使用的固定最终诊断。新实验的收益以完整测试为准，不预先宣称改善。
