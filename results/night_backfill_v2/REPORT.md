# 三种保留方法：夜间评测补全

固定已选 best 权重，测试不再选模；无新训练。每格为 R@1 / R@5 / R@10（%）。

| 协议 | Query | 原始模型 | Mixed 日夜混合 | Mixed＋相机退化 |
| --- | ---: | ---: | ---: | ---: |
| msls_val_night_day | 55 | 85.4545 / 94.5455 / 96.3636 | 89.0909 / 94.5455 / 98.1818 | 90.9091 / 100.0000 / 100.0000 |
| msls_test_night_day | 309 | 36.2460 / 44.6602 / 46.9256 | 39.1586 / 50.1618 / 53.0744 | 37.8641 / 47.5728 / 51.4563 |
| svox_test_night | 823 | 87.6063 / 93.8032 / 95.3827 | 93.6817 / 97.2053 / 98.6634 | 94.1677 / 98.4204 / 98.9064 |
| darkdriving_paired_diagnostic | 9530 | 9.0031 / 34.0189 / 53.3158 | 11.4166 / 42.2350 / 62.9591 | 11.4690 / 42.6967 / 63.4942 |
| robotcar_gps_night_diagnostic | 22495 | 86.2192 / 92.7895 / 94.8522 | 93.0607 / 96.9638 / 97.9862 | 92.9362 / 96.8215 / 98.1240 |
| robotcar_seasons_public_night | 197 | 90.8629 / 97.4619 / 98.4772 | 94.9239 / 98.9848 / 99.4924 | 97.4619 / 99.4924 / 100.0000 |
| robotcar_seasons_public_night_rain | 224 | 92.8571 / 98.2143 / 99.1071 | 96.4286 / 98.6607 / 99.5536 | 96.4286 / 98.2143 / 99.5536 |
| gardenspoint_night_day_right | 200 | 88.5000 / 98.0000 / 99.5000 | 89.5000 / 97.0000 / 98.0000 | 92.0000 / 99.0000 / 99.0000 |
| gardenspoint_night_day_left | 200 | 81.0000 / 98.0000 / 100.0000 | 81.5000 / 96.5000 / 99.0000 | 84.0000 / 98.0000 / 99.0000 |
| sped_mixed_illumination_diagnostic | 607 | 74.7941 / 84.8435 / 86.9852 | 75.7825 / 84.8435 / 87.4794 | 76.7710 / 86.9852 / 88.3031 |
| tokyo105_night | 105 | 83.8095 / 92.3810 / 93.3333 | 87.6190 / 94.2857 / 94.2857 | 84.7619 / 93.3333 / 94.2857 |

Night 与 Night-rain 为两个独立条件；Gardens 的两行使用同一批夜间查询、不同白天图库。
RobotCar GPS、RobotCar-Seasons public-train、DarkDriving 均为明确命名的诊断协议，不冒充官方定位 test。
SPED 为补充的混合光照诊断，没有验证过的逐图夜间标签，不计入严格夜间数据集总数。
同一协议下三方法共享同一冻结教师图库、原始测试图片、FP32 与关闭 TF32 的精确检索。
已有结果仅在权重、输入顺序、真值、缓存完整字节和预测命中核验后复用。

未能给出 Recall 的项目及原因：
- Aachen-Day-Night official night test：No public query pose / verified query-to-gallery retrieval positives
- RobotCar-Seasons official night / night-rain test：Test query poses withheld; public-train diagnostic evaluated separately
- NightCity / BDD10K / MUSES：Night images exist but no verified place-retrieval ground truth
- SVOX val：Local validation queries are Mapillary daytime views, with no verified night-query subset
- NCLT / 17Places：No complete local image/ground-truth bundle
- Boreas / Nordland：No verified night-query labels in the available local traversals

状态：complete。
