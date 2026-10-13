# pointwam-repro 当前状态

## 2026-10-13 远端下载仍在继续，已下载部分开始并行重建

EgoDex 下载进程还在跑。part2 已核对 95004/95125 个包，121 个失败所以这一卷仍是 INCOMPLETE；part1 正在下载。GPU3 上原来的重建没有停。已经下完、但 GPU3 还没走到的 TRAIN 记录，改由空闲的 GPU0–2 处理，产物写回同一套 reconstruction 目录，并留出距离以免和 GPU3 正在写的那一条冲突。第一批 49 条已完成并核对过深度和窗口文件。三个追加进程的预算是 3 天，状态在 `/mnt/ugreen_nas/storage/oyx_storage/egodex-aux-reconstruct-20261013-r1/gpu{0,1,2}-status.json`。这些重建结果的 `training_allowed` 仍是 false，还不是 PointWAM 训练集。

## 2026-10-13 冻结编码器上的一步优化已经跑通

GPU0 上用这条合格 clip 的全部 5848 个当前有效场景点，冻结 Mosaic3D 和 Recap-CLIP 提供特征，随机初始化的 Volt-H 按 Table 4 走了一步 AdamW。损失 7.903，裁剪前梯度范数 5.592，两者有限。峰值 13028 MiB，优化步 0.604 秒。分步标准化只用了这一条 clip，不是论文的全数据统计。`training_allowed` 仍为 false。60k 和 global batch 192 没有启动。远端 GPU3 没有动。详见实验卡。

## 2026-10-13 冻结编码器的小规模前向已经接通

本机从 hf-mirror 经代理下完 Mosaic3D `sc+ar+sc+++ma+st.ckpt`，aria2 平均 1.8 MiB/s，约 23 分钟，文件 2736205302 字节。压缩包头尾完整，`torch.load` 后 `net.*` 与本地 PPT/SpUNet 键完全一致。Recap-CLIP 此前已下完。

GPU0 上对本地合格 clip 取 1024 个场景点，冻结 Mosaic3D 给出 768 维点特征，冻结 Recap-CLIP 给出 14 个 1024 维 token。随机初始化的 Volt-H 在 bf16、`no_grad` 下输出有限，峰值 6479 MiB，整段 22.6 秒。这是接口检查，不是运动预测结果，也不是 60k 训练。`training_allowed` 仍为 false。远端 GPU3 没有动。详见实验卡。

## 2026-10-13 论文 Volt-H 可以在一张 3090 上走完一步

官方代码仍未公开。可训练部分已按 Appendix B.3 和 Table 4 写好：1 cm 稀疏 stem、5³ patch、32 层宽度 1280 的 Volt-H、轴向 RoPE、解码头、零初始化融合残差、分步标准化和运动加权损失。参数量 655,527,092，论文预训练写的是 651M。差出来的部分来自论文没有写死的手部 MLP、残差块和 detokenizer 中间层。

本机空闲 GPU0 上，12000 个随机点、0.8 m 范围、bf16、AdamW 1e-4、裁剪 5，一步完成。损失和梯度有限，峰值 12776 MiB，步耗时 1.224 s。场景特征和文本特征是 768 / 1024 维随机张量，不是 Mosaic3D 或 Recap-CLIP。因此这只说明网络能跑，不是预训练结果。60k、global batch 192 没有启动：按这一步的速度，单卡累积到 batch 192 大约要数月。数据规模仍按用户要求保持现状，不用 115 万 episode。远端 GPU3 的全量重建没有动。

下一步是接入冻结编码器，再在已有 clip 上做一步真实前向。不要把缩小版两层模型继续加长。详见实验卡。

## 2026-10-13 远程 clip 合同通过，训练未开始

目标仍是先复现人类预训练。官方代码和 checkpoint 未公开。2026-10-13 15:20 左右只读登录 `deepcybo-4-4090-48`，没有写远端、没有停进程。

GPU3 被 `reconstruct_egodex_queue.py`（pid 360908）占用，输出目录是 `/mnt/ugreen_nas/storage/oyx_storage/egodex-full-20261012-r1`。状态为 `RECONSTRUCTING`：完成 4563、失败 17479、已准入窗口 37938/39298。这是全量重建，不是下面这 48 个 clip。GPU0–2 当时空闲。

`pointwam-data-expansion-20261011-r1` 的 48 个合格 clip 全部通过本地复述的合同：31/30 帧、每手 10 点、z-up、源帧间隔 3、位移等于相对第 0 帧的差（834 万对最大差 0）、至少一只完整手。场景点数 2856 / 6987 / 12000（最小/中位/最大）。手部末端有效点位移中位数约 79.2 mm。schema 为 `ref2dex.pointwam-data-smoke.v1`，包内和 manifest 的 `training_allowed` 都是 false。fit/dev 为 23/25。摘要见 [summary.json](../../../../outputs/pointwam-repro/clip-contract-audit-20261013-r1/summary.json)。

本任务现在有缩小版 1 cm 体素、5³ patch、运动加权平方误差和两层 transformer。它用 XYZ/RGB MLP 代替冻结 Mosaic3D，用一个可学习 token 代替 Recap-CLIP，不是 Volt-H。6 个本地测试通过。没有在这 48 个 clip 上训练。

## 2026-10-13 缩小 patch 模型短训：手能拟合，场景和 dev 变差

GPU0 上完成 `P-20261013-pointwam-repro-patch-fit`。r1 因无效点 NaN 污染损失而作废。零初始化 1000 步后，fit 手部 EPE 54.8→29.5 mm，fit 场景 9.0→15.1 mm，dev 手部 69.3→87.1 mm。按静态损失平衡两项后再跑一遍，模式相同。结论 UNCLEAR：不要加长这套替代网络，也不要据此去接 Mosaic3D。GPU3 的全量重建没有动。详见实验卡。

下一步：先修场景与手的共同改进，再考虑冻结编码器和全分辨率点云。
