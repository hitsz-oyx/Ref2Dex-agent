# PTv3-Shared-WAM

实现依据是 [ref1.md](docs/user/ref/ref1.md)。实际模型、张量合同、损失与执行语义见
[架构.md](docs/user/架构.md)。本 Task 单独实现共享预测器，不修改其他 Task 的模型或训练入口。

## 入口

使用 `/home2/wyy/miniconda3/envs/graspenv/bin/python`。已有三域数据、TRAIN 统计和
PW step50000 的默认路径在 `src/ptv3_shared_wam/training.py`；CLI 可以显式覆盖。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/train_pretrain.py \
  --smoke --gpu 0 --output outputs/ptv3-shared-wam/<fresh-pretrain-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/train_robot.py \
  --smoke --gpu 0 --init-shared outputs/ptv3-shared-wam/<pretrain-run>/step4.pt \
  --output outputs/ptv3-shared-wam/<fresh-robot-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_native.py \
  --checkpoint outputs/ptv3-shared-wam/<robot-run>/step4.pt --controls 4 --execution-steps 4 \
  --gpu 0 --output outputs/ptv3-shared-wam/<fresh-native-run>

/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/audit/audit_native.py \
  --native outputs/ptv3-shared-wam/<native-run>/native
```

非 smoke 运行须先建立 Task-local experiment card，分配全局 seed、说明问题/决策/预算，
通过 `--experiment-card` 指定；训练 seed 来自卡片第一个 seed，rollout seed 必须列在卡片内。
engineering checkpoint 不会静默进入科学微调或完整物理评估。

预训练和机器人微调均支持 `torchrun`。显式设置空闲设备，例如
`CUDA_VISIBLE_DEVICES=0,1 .../torchrun --standalone --nproc_per_node=2 .../train_pretrain.py ...`。
`batch_size` 是每 rank 数量。`--resume` 恢复模型、优化器、步数和每 rank 随机状态；
保持 stage/config/数据路径和 world size 一致。检查点采用唯一 `stepN.pt`，不覆盖历史文件。

机器人初阶段默认无 human replay；通过 robot config 的 `replay_fraction` 开启后续混合阶段。
人类样本只计算手/场景损失，不生成或使用机器人动作标签。

## 验证

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest src/task/ptv3-shared-wam/tests -q
```

CPU 合同测试使用可微场景接口 fixture；真实稀疏 PTv3、权重迁移和机器人梯度由
GPU 工程 smoke 验证。工程 smoke 不能说明预训练已有效、策略能抓取或 Cm 有收益。

当前验证记录：14个合同测试通过；双卡预训练与双卡robot+25%human replay各4步通过；
从step2恢复至step4的参数最大差为 `3.47e-7`。
原生工程入口默认4步；`--smoke --controls 16`允许验证完整8步buffer的两次查询，
仍不评价抓取效果。

| 最终工程检查 | run_id |
|---|---|
| 双卡共享预训练 | `engineering-ddp-pretrain-20261013-r2` |
| 双卡机器人＋human replay | `engineering-ddp-robot-replay-20261013-r2` |
| 连续更新 / checkpoint恢复 | `engineering-resume-original-20261013-r1` / `engineering-resume-restored-20261013-r1` |
| 原生K=1 / 4 / 8 | `engineering-native-k1-20261013-r1` / `engineering-native-k4-20261013-r2` / `engineering-native-k8-20261013-r1` |

上述产物位于仓库 `outputs/ptv3-shared-wam/`，各原生run的 `native/independent_audit.json`
复算命令、PD、自己的H4/R-state、计划goal和chunk索引。
早期r1预训练/机器人checkpoint属于scene读出定稿前的工程接线，保留原日志，
不作为最终模型的学习效果证据。第一次DDP robot replay被GPU0的外部任务触发资源停止；
已换GPU3/4完成r2，没有停止外部进程。

Residual PPO、旧 counterfactual World 和旧架构 matched 科学比较是后续实验，
本次没有启动，也没有为目录完整性创建空实验卡。
