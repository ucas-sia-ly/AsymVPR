# 三种保留方法的夜间评测补全

本轮仅包括原始、Mixed、Mixed＋相机退化。使用各方法已有的固定 best，归档方法不纳入；
新增测试不参与训练、调参或重新选模。Mixed＋相机退化已经完成 10 轮 / 9,080 次更新，
其真实训练记录与曲线见 [TRAINING.md](../results/night_backfill_v2/TRAINING.md)。

固定权重、源评测 SHA 与范围保存在 [night_backfill_v2.json](../configs/night_backfill_v2.json)。
独立入口为 `python -m reproduction.night_backfill`，输出到 `results/night_backfill_v2/`；
不改写此前 `mixed_night_suite_v1` 的原始结果。

| 协议 | Gallery | Query | 用途与真值 |
| --- | ---: | ---: | --- |
| MSLS val night→day | 18,841 | 55 | 原 night 标签，非 panorama，同城 UTM ≤25m，自定义子集 |
| MSLS test night→day | 36,362 | 309 | 同上；不是完整官方 all/subtask 分数 |
| SVOX test-night | 17,166 | 823 | test 划分，UTM ≤25m；与选模开发集分开 |
| DarkDriving | 9,530 | 9,530 | 严格同名配对诊断；附近帧可能成为假负例 |
| RobotCar GPS night | 21,006 | 22,495 | 既有 GPS 1m 采样，精确图像时间戳关联，UTM ≤25m，诊断 |
| Tokyo105 night | 75,984 | 105 | 官方 MAT 的 315 查询中 CSV `N` 标签，固定最终诊断 |
| RobotCar-Seasons public Night | 20,862 | 197 | public-train rear 位姿，完整 49 模型参考图库，3D ≤25m，诊断 |
| RobotCar-Seasons public Night-rain | 20,862 | 224 | 独立夜雨条件；与上一项共享同一图库 |
| Gardens Point night→day_right | 200 | 200 | 发布者原图与 VPR-Bench 的帧对应 / ±2 帧容差 |
| Gardens Point night→day_left | 200 | 200 | 同一夜间查询，额外包含横向视点变化的白天图库 |
| SPED mixed illumination | 607 | 607 | 补充混合光照诊断；没有验证过的逐图 night 标签 |

共 11 个协议 × 3 种方法 = 33 项。Gardens 的两个协议不是两组独立查询；SPED 不计入严格夜间子集。
所有方法共享各协议的同一冻结 BoQ 教师图库，以学生编码原始 query，使用 FP32、关闭 TF32、精确内积检索。
测试不额外加入相机退化。Recall@1/5/10 为百分数，同时保存每张 query 的命中标记与 top-10。

## 新增真值来源

RobotCar-Seasons 使用用户 `WORKSPACE/dataset/RobotCar-Seasons` 中完整的公开元数据，
复用 WeatherVPR 已审核过的 49-model 严格解析器，仅生成一次封存协议 JSON。
公开 query 的矩阵为 camera-to-world，中心取最后列；参考 COLMAP 为 world-to-camera，中心取 `−Rᵀt`。
参考图库覆盖所有 49 个 aligned 模型，包含 left/rear/right；无朝向或可见性过滤。
逐模型元数据、解析器源码 SHA 与完整正例列表保存在 `configs/evaluation_sources/robotcar_public/`。
来源：[官方 v2 说明](https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/README_RobotCar_v2.md)。

Gardens Point 使用[发布者 Zenodo 原始 ZIP](https://zenodo.org/records/4590133)，先核验发布 MD5，
再按原始 `day_left/day_right/night_right/Image000.jpg..Image199.jpg` 解出所有 600 张图，JPEG 字节保持原样。
原本的本地平铺版本未通过原始字节匹配，因此没有猜测其重命名对应关系。
正例来自 [VPR-Bench 作者发布的 ground_truth_new.npy](https://github.com/MubarizZaffar/VPR-Bench/blob/77ad30f3684476717188c67d7f3f66be19f832af/datasets/GardensPointWalking/ground_truth_new.npy)，
固定到 commit `77ad30f3684476717188c67d7f3f66be19f832af`。
原文件的少量边界正例超出 0..199，本协议显式与实际 200 个参考索引求交，记录裁剪数量；
不凭相似度生成正例，不把帧容差描述成米制距离。来源及图片 SHA 见 `configs/evaluation_sources/gardenspoint/`。

协议重建只做 CPU 数据核验：

```bash
/home/admin123/miniconda3/envs/boq/bin/python -m reproduction.prepare_night_backfill
/home/admin123/miniconda3/envs/boq/bin/python -m reproduction.night_backfill --manifest-only
```

新环境需先取上述已注明 MD5 的原始 ZIP 到 `artifacts/benchmark_sources/GardensPointWalking.zip`。
协议 JSON 绑定公开源文件和图像版本；来源变化时拒绝静默复用。

## 已完成结果复用与资源

旧结果仅在权重 SHA、教师、FP32/TF32、图库/查询有序列表、文件状态、正例和元数据全部一致时复用。
完整缓存 SHA 和 top-10 再核验，逐 query 重算命中与 Recall。Tokyo 从已有完整 top-10 按可信夜间标签重算，
标为已有推理，不伪装成本次重新提取。新增协议真正运行三个模型；相同图库和相同查询通过特征身份共享缓存。

Trainroom 任务 `5c702969e8cc`，中文名称“三方案夜间补测”，日志为 `results/night_backfill_v2/logs/metrics.jsonl`。
进度轴为已处理的“方法×协议”，不是优化器更新。提取图片数和耗时按阶段单独记录。
本次使用 teacher/student batch 128、4 workers、GPU 预留 4 GiB、CUDA 分配上限 32 GiB，
主机 MemoryHigh=6 GiB、MemoryMax=8 GiB、SwapMax=0。中断可按相同配置恢复已提交的特征行。

```bash
python3 /home/admin123/.codex/skills/local-training-terminal/scripts/trainroom.py launch \
  --cwd /home/admin123/github/AsymVPR \
  --name 'AsymVPR · 三方案夜间补测 · Mixed与相机退化 · 33项' --total-steps 33 \
  -- /usr/bin/env ASYMVPR_EVAL_MODULE=reproduction.night_backfill \
  ASYMVPR_EVAL_MEMORY_HIGH=6G ASYMVPR_EVAL_MEMORY_MAX=8G \
  /bin/bash reproduction/eval_launch.sh --registry configs/night_backfill_v2.json \
  --batch-size 128 --teacher-batch-size 128 --workers 4
```

同一任务仍在运行时只查看已有任务，不另起一份。完成后生成绑定汇总和报告 SHA 的 `complete.json`。
代码、配置或真值变化时，改用新输出目录。

## 未能计算 Recall 的范围

Aachen 官方夜间 test 和 RobotCar-Seasons 官方 Night / Night-rain test 没有公开 query 位姿或可信检索 positives；
本轮没有用预测结果充当真值。NightCity、BDD10K、MUSES 的夜图没有验证过的地点检索真值；
NCLT/17Places 缺完整本地数据；本地 SVOX val 为 Mapillary 白天 query，Boreas/Nordland 未提供验证过的夜间标签。
各项原因与已覆盖协议都写入 [coverage.json](../results/night_backfill_v2/coverage.json)。

结果见 [统一报告](../results/night_backfill_v2/REPORT.md) 与 [完整汇总](../results/night_backfill_v2/summary.json)。
只按相同协议比较，不把配对诊断、开发集、公开 train 诊断与官方 test 混为一个“总体夜间 Recall”。
