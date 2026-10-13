# pointwam-repro

按论文复现 PointWAM 的独立任务。这是 paper-faithful reimplementation，不是官方代码回放。作者项目页在 2026-10-13 仍没有可引用的训练仓库或 checkpoint：[PointWAM](https://chrockey.github.io/PointWAM/)，论文 [arXiv:2610.02840](https://arxiv.org/abs/2610.02840)。

本任务不改 `goal-point-wam` 的 actor、World 或 Chunk-R，也不把现有远程 EgoDex 作业当成 PointWAM 预训练。

## 要复现的合同

人类阶段输入是当前彩色场景点、当前语义手点、语言指令。输出是相对当前坐标的场景点位移和每手 10 个语义点位移。场景点和手点在同一米制、z-up 坐标系。人类窗口是 31 帧、约 1 秒、30 个未来位移。Mosaic3D 与 Recap-CLIP 冻结，Volt 轨迹主干和两个确定性平方误差头从头训练。机器人阶段在此之后：预报手轨迹加当前机器人状态，经 learned retargeter 得到 action chunk。论文预算是约 115 万人类 episode、每阶段 6 万步、global batch 192、4×B200。第一轮不按这个预算启动。

## 可直接复用

数据合同和小型工程入口在 `cm-pointflow-effect-pretrain`，只借用，不把那边的替代网络称为 PointWAM：

- `src/oakink_wm/pointwam_core.py`：10 点手、clip、mask、联合损失接口
- `src/oakink_wm/pointwam_data.py`：EgoDex 读取
- `tools/run/prepare_pointwam_pilot.py`：CoTracker3 + VGGT-Omega 的小规模轨迹生成
- `tools/run/qualify_pointwam_train.py`：官方 TRAIN 小子集资格检查
- 已通过数值门的两段 pilot 与 12 个 TRAIN clip 资格记录，见该 Task 的 `P-20261011-pointwam-data-pipeline` 与 `P-20261011-pointwam-train-qualification`

`PointWorldWAM` 是 GRAB 上的 PointWorld 迁移，接口不同，不作为本任务主干。`goal-point-wam` 的物体 goal、PTv3 和动作条件 World 也不是这篇论文的输入。

## 当前进度

2026-10-13 论文尺寸的 `PointWAMForecaster` 已能在本机 GPU0 上对 12000 点做一步 bf16 反传，峰值约 12.8 GB。同一天，冻结的 Mosaic3D 和 Recap-CLIP 已接进这个网络：1024 点的 `no_grad` 前向有限，整条 clip 的 5848 点也按 Table 4 走完一步 AdamW，损失和梯度有限，峰值 13028 MiB。轨迹头仍是随机初始化，标准化统计只来自这一条 clip，不能当成论文预训练结果。60k、global batch 192 没有启动。缩小版两层模型的短训仍然是 UNCLEAR，不再加长。

`src/pointwam_repro/` 里缩小 forecaster 仍保留，用来对照早期实验。论文模型、冻结编码器合同和标准化统计是后面加上的。实验卡：

- `docs/experiments/probes/P-20261013-pointwam-repro-patch-fit.md`
- `docs/experiments/probes/P-20261013-pointwam-repro-volt-forward.md`
- `docs/experiments/probes/P-20261013-pointwam-repro-frozen-forward.md`
- `docs/experiments/probes/P-20261013-pointwam-repro-frozen-step.md`

```bash
PYTHONPATH=src/task/pointwam-repro/src /home2/wyy/miniconda3/envs/graspenv/bin/python \
  -m pytest src/task/pointwam-repro/tests/test_repro_contract.py -q
```

当前判断见 [docs/STATE.md](docs/STATE.md)。
