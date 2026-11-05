# AsymVPR 主配置复现报告

DINOv2-B + BoQ 图库 → EfficientViT-B2 + BoQ 查询。独立实现；作者仓库本次核对返回 404。

## 训练来源

复用本机先前完成的 15 epochs、124,110 steps 训练。本次没有重复训练。
最佳权重由 MSLS validation R@1 选择：epoch 9、step 74,466。
权重 SHA256：`d051f6cec174614a7fed1391624420308f31cac6aa59782530f582a01eacb54a`。

历史训练最佳 MSLS R@1/5/10：92.1622 / 96.2162 / 96.3514%。
下表仅使用本次独立评测输出；空项表示尚未产生结果。

## 独立评测

本次评测进程状态：**completed**。尚无结果的数据集保持空缺。

| 数据集 | 本次 R@1 % | 论文 R@1 % | 差值（百分点） | 本次 R@5 % |
| --- | ---: | ---: | ---: | ---: |
| MSLS val | 92.1622 | 92.3 | -0.1378 | 96.2162 |
| Pitts250k test | 94.7947 | 95.4 | -0.6053 | 98.2971 |
| Tokyo24/7 | 91.7460 | 92.7 | -0.9540 | 96.1905 |
| Nordland | 78.5145 | 74.6 | +3.9145 | 88.9130 |
| AmsterTime | 47.6036 | 48.6 | -0.9964 | 69.2933 |

Gallery 始终由冻结教师编码，query 由同一个最佳学生权重编码；FP32、TF32 关闭，分块穷举检索。
MSLS 使用 740 个有真值查询；它同时是选模验证集，不能视作独立 test 成绩。

### 实际评测协议

- **MSLS val**：18,871 references / 740 queries；`{"split": "validation", "positives": "official pIdx, 25 m geographical threshold", "query_selection": "official qIdx, queries with known positives", "all_query_count": 747, "selected_query_count": 740}`。
- **Pitts250k test**：83,952 references / 8,280 queries；`{"split": "pitts250k_test", "positives": "official GSV-Cities/SALAD ground truth, 25 m", "source_mat_audit": "all image IDs/order and all 25 m positive sets match exactly"}`。
- **Tokyo24/7**：75,984 references / 315 queries；`{"split": "Tokyo24/7, published 315-query subset", "positives": "UTM coordinates encoded in official paths; Euclidean distance <= 25 m", "resolution": "original VGA, deterministic resize to configured input size"}`。
- **Nordland**：27,592 references / 2,760 queries；`{"split": "VPR-Bench sampled summer queries / winter references", "positives": "reference sequence frame +/- 3 inclusive", "query_frame": "query ordinal * 10", "protocol_variant": "paper", "published_gt_differing_rows": 2760, "image_layout": "original VPR-Bench release ref/query"}`。
- **AmsterTime**：1,231 references / 1,231 queries；`{"split": "all 1231 published pairs", "references": "new images", "queries": "old images", "positives": "image_pairs.csv old_num -> new_num; no coordinate proxy"}`。

Nordland 协议诊断：对同一组预测改用发布的 ±1 帧真值，R@1/5 为 72.2101 / 86.3406%。主表按论文文字采用 ±3 帧；容差对分数影响很大，不能把高于论文的差值直接解释为方法改进。

## 本机推理计时

| 模型 | 参数量 | batch=1 中位延迟 ms |
| --- | ---: | ---: |
| DINOv2-B + BoQ | 95,206,560 | 9.061 |
| EfficientViT-B2 + BoQ | 22,275,984 | 3.502 |

同机网络前向中位延迟比为 2.59×。
NVIDIA GeForce RTX 4090，FP32、322×322，预热 10 次，计时 50 次，CUDA 同步 wall clock。
计时使用重复的合成 GPU 输入，不含读图、预处理、传输或检索；不能直接与论文的硬件计时比较。

## 复现边界

- 本机 GSV-Cities 为 529,506 图像、64,394 地点；论文写约 560k / 67k。
- 论文未公开完整增强、插值、采样、协方差估计和 checkpoint 选择细节；实际选择详见 README。
- Tokyo 使用本机原始 VGA PNG，AmsterTime 使用原始现代 PNG / 历史 JPG；作者评测图像编码未公开。
- Nordland 默认使用 VPR-Bench 发布，冬季 references、夏季 queries，query 每 10 帧采样；具体正匹配容差见实际评测协议。论文写 ±3 帧，但没有明确季节方向。
- 尚未复现 SALAD、MobileViTv2、全部对比方法及消融实验。

有序图像清单的哈希、元数据/真值/模型/特征哈希和预测 Top10 均保存在 evaluation 目录。
训练完整历史来源见 training_history.json；训练和计时均保留 Trainroom 原始日志。
