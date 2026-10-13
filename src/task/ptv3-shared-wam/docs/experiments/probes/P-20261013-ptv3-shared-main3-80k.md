---
schema: ref2dex.probe.v2
probe_id: P-20261013-ptv3-shared-main3-80k
experiment_id: P-20261013-ptv3-shared-main3-80k
date: 2026-10-13
task: ptv3-shared-wam
branch: main
git_commit: 17dd9349
claim_id: C1
hypothesis_family: HF-ptv3-shared-task-forecaster
probe_index_in_family: 1
seed_pool: probe
seeds: [100]
decision_changed_if_positive: transfer the trained shared forecaster into airplane full-pair adaptation
decision_changed_if_negative: inspect the forecast contract and data before spending on robot adaptation
status: PLANNED
run_id: pretrain-main3-ddp4-80k-20261013-r1
---

# 四卡三域共享预测器预训练：80,000 updates

Result: PLANNED. 用户明确要求直接启动四卡大规模训练；尚无学习效果或抓取结论。
Decision: 固定ref1_2的共享goal架构，完成负载检查后启动80,000次优化器更新。

Class: Decision. 服务于自主灵巧操作的预测器基础，而不直接验证Cm增益。具体问题是：
该共享架构能否在三域实际手物运动上学到手轨迹和超过endpoint插值的中间场景预测。
正向信号支持下一阶段airplane完整forecaster+R适配；无手轨迹学习或中间预测收益时，
先查本Task数据/合同，不把human预测误差当作机器人抓取成功率。

## Decision Note

- 用户已授权四卡、80,000步及较大batch，本次按三域预训练解释。
- 已有20项架构合同测试及真实PTv3/robot/native短smoke；长期运行调整后22项测试通过。
  这些仅证明接线，不证明学习效果。
- 选择每卡128、global batch512；单卡128负载检查通过，峰值分配显存约12.8GiB，
  稳定约2.15秒/步，比每卡64的样本吞吐约高20%；四卡DDP四步也已通过，约230–245样本/秒、显存约14–16GiB/卡。
- 不扩展架构或数据，不改变核心claim。单seed长期训练仍是探索性训练，不是matched Validation。
- 初步预算约2–3天，硬上限7天/四卡；4GB单run、400GB共享产物预算保持不变。
  非有限loss/梯度、OOM、外部GPU进程、源码变化、30分钟无进展或资源上限触发停止。
- 成功后保留checkpoint进入airplane适配设计；失败保留日志和最近恢复点，先定位实际问题。
  没有需要追加的资源/权限授权。

## 固定协议

- 配置：[pretrain_80k.json](../../../configs/pretrain_80k.json)。80,000是optimizer updates，
  不含额外梯度累积；每步全局512样本，合计40.96M次有放回窗口抽样。
- 物理GPU4/5/6/7，NCCL DDP；启动前确认空闲。模型全量更新，bf16 autocast。
- OakInk2/GRAB/ARCTIC原TRAIN/dev划分，来源权重5:2:2和原category sampler。
  保持TRAIN-only统计、H4、30Hz、horizon24、每物体32点。
- 仅从已有PW step50000迁移场景编码器，不加载四步engineering预测器作为科学初始化。
  原PTv3结构不变；256 scene、22 hand、32 goal共同进入4层128D共享Transformer。
- 目标：分别归一化的hand/scene MSE，既有motion weighting不变。
  AdamW；PTv3/scene lr1e-5，其余1e-4，weight decay0.01，clip1。
  增大batch时不自动线性放大学习率，也不新增scheduler。
- 主种子100，rank初始化100–103；每步按主种子/step/rank派生独立抽样子流。
  固定dev panel seed104，共384窗、每source128；rank不是独立科学重复。
- 每1,000步验证：各source hand first8/all24；scene h1/first8/middle(9–16)/h24/all24、
  target/non-target、moving/static，以及goal-only线性插值基线。
- 主要读hand first8/all24跨source的变化。scene重点读first8/middle及non-target，
  不把目标h24近零当作独立动力学收益；无有效样本的类别报告null和count0。
- 每2,000步保存恢复点；保留step1、20k/40k/60k/80k及最近一个恢复点。
  删除范围仅限该run的登记表中被替代的非里程碑恢复点，完整验证曲线与其他run不删除。
  最多同时约6份617MB文件（含写入暂存），仍低于4GB。
- 每20步打印loss、GPU、step time、样本吞吐和ETA；监督进程每30秒记录GPU快照。
  首次梯度审查用实际小batch，避免保留FP32图限制bf16训练batch。

## 运行与追溯

实际运行commit、输入路径、worker PID、config和初始化记录以run的manifest为准。
启动后回填此卡。产物：`outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r1/`；
launcher日志、GPU采样和scratch位于同名项目`tmp/`目录，训练目录提供对应链接。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/ptv3-shared-wam/tools/run/launch_pretrain.py \
  --detach --gpus 4,5,6,7 \
  --config src/task/ptv3-shared-wam/configs/pretrain_80k.json \
  --experiment-card src/task/ptv3-shared-wam/docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md \
  --output outputs/ptv3-shared-wam/pretrain-main3-ddp4-80k-20261013-r1
```

工程负载记录：`engineering-batch64-20261013-r1`、`engineering-batch128-20261013-r1`；
四卡检查（GPU0/1/2/3）：`engineering-ddp4-batch128-20261013-r1`。均为四步，不能据此形成训练效果结论。

## Limitations / future evidence

单seed、三域dev曲线不替代匹配对照、多seed或机器人闭环。未来需评估完整airplane适配、
固定pair下K-step执行与最终Cm增益。视频/RGB输入、EgoDex、旧架构matched比较和
任意goal反事实正确性不属于本轮，当前不占用四卡预算。
