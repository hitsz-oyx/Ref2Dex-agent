---
schema: ref2dex.probe.v2
probe_id: P-20261023-tau-checkpoint-execution-ab
experiment_id: P-20261023-tau-checkpoint-execution-ab
date: 2026-10-23
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-tau-deployment-execution
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
decision_changed_if_positive: promote step1223 if net successes increase without increased safety aborts; schedule component decomposition
decision_changed_if_negative: do not accept tau by EPE alone; locate actual consumed forecast segment if success regresses
status: RUNNING
run_id: tau-checkpoint-ab-20261023-r1
---

# 新旧 τ 的部署条件配对执行

| 结果解读（运行前固定） | 含义 | 决定 |
|---|---|---|
| 新 ckpt 净成功数增加，安全中止不增 | 预测改善在此固定面板可转为执行收益 | 新 ckpt 定为主线 τ，排新 ckpt 成分分解 |
| 净成功数不变、死因/阶段改变 | 改善或退化只到执行链部分环节 | 结合 contact/stage/首触限/目标误差定位，不称整体收益 |
| 净成功数下降 | 此面板 EPE 与执行需求解耦 | 停止以 EPE 作为 τ 验收，查实际消费段 |
| 成功增加但安全中止增加 | 收益伴随安全回退 | 不满足主线替换门，保留两项独立结果 |

Decision Probe：区分同一部署条件下，no-goal/no-event 训练的预测改善是否改变完整执行结果。
最便宜的可判别方法为既有20 reset × 两个checkpoint，无训练、无新模型或baseline。

面板选取：v11 coverage40 registration 原顺序前20条 p000–p019，恰好20物体各1条；
保留原reference/native seeds100–119/model seeds180–199，不按结果选样。
用户未指定更精确路径，运行前已提出此明确假设。该面板已用于开发，非未见泛化验证。

两臂重跑，均显式 `goal=None`、`event=[None]`，真实模型入口逐次断言 goal/event 全零、valid=False。
历史 runner 实际传 scheduler goal/event，故不复用历史成功数作为本轮基线。
执行器、相位机、R权重、native host和评分器共同冻结为v11 g4/runtime-code；
两臂共同使用当前v6模型加载/条件适配包，保留冻结的staged_progress.py。
此共同适配支持新checkpoint的goal/event policy，不更换模型结构；checkpoint配置差异已核对，
仅条件policy和训练元数据变化。主指标仍为600controls末态、实测hold、尺寸阈值目标误差与无安全中止。
原安全门不变，接触未认证是末态失败分类，不新加即时abort门。

旧：`outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt`。
新：`tmp/t38-r2/training/step001223.pt`（1223次更新最终态，非挑best）。
同一GPU上每对交替old/new顺序，每episode重新设同一seed、启动fresh native进程。
保存九字段initial.npz并与配对臂及原v11初态逐数组比；不宣称隐藏PhysX缓存已认证。

预算：GPU4/5/6/7各5对，最多24000controls、0updates；worker7200s、supervisor7500s；
预计<1GiB、硬上限2GiB、可用<20GiB停，运行前盘约130GiB。无视频、无checkpoint复制。
Python graspenv3.8.20/Torch2.4.1+cu121；native venv3.8.10/Torch2.0.1+cu118。
实际环境、commit、GPU/显存/利用率、controls与墙钟写入各manifest和supervisor。

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python src/task/wam-rl-policy-v3/tools/run/run_tau_checkpoint_ab.py --output outputs/wam-rl-policy-v3/tau-checkpoint-ab-20261023-r1 --detached
```

输出在上述目录：registration、shared-runtime、worker.diff、每GPU manifest与逐回合轨迹。
将报告逐臂成功、joint abort/contact not certified/table penetration、首触限全部关节（同tick可并列）、
stage_achieved逐项/最高阶段、final_goal_error_m分位数及配对逐例差。
正式结果尚未产生；不提前替换其他入口默认checkpoint。

局限/未来证据：固定20面板是Probe；模型随机/物理数值敏感性未做重复，不能当总体因果/泛化结论。
成分分解仅在符合用户指定的成功与安全分支后排期，本轮不运行第三臂。
