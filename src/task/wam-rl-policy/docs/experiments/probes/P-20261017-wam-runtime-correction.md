---
schema: ref2dex.probe.v2
probe_id: P-20261017-wam-runtime-correction
experiment_id: P-20261017-wam-runtime-correction
date: 2026-10-17
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-wam-runtime-correction
probe_index_in_family: 1
seed_pool: probe
seeds: [245, 249]
decision_changed_if_positive: treat the earlier WAM adapter drop as implementation-confounded and continue with a matched corrected adapter comparison
decision_changed_if_negative: stop this adapter recipe and return to WAM goal semantics or a matched trainable-module design
status: RUNNING
run_id: corrected-wam-{smoke,eval,train}-remote-20261017-r1
---

# 修正 WAM adapter 的执行一致性与逐环境 history

Result: 运行中。该 Probe 只判断两个已定位的 runtime 问题是否足以改变旧 adapter 负结果
的解释，不形成 WAM 表征的正式有效性结论。

## Decision Note

当前需要决定的问题是：旧 adapter 的 `1/128`、`2/128` 和有界残差退化，是否至少有一部分
来自确定性的评估重复加法和跨环境 history 清空，而不是来自 WAM 表征本身。

关键证据是旧生成 runtime 在 `act`/`evaluate` 各加一次 residual，却在 `act_inference`
再次加一次；wrapper 还用 `dones.any()` 清空整批 history。最便宜的判别方法是先修正生成器，
做源代码路径计数、非零 adapter 动作均值一致性审计和旧 checkpoint 确定性重评，再决定是否
投入短训练。

## Protocol

1. `official_unidex_ppo.py` 的 adapter residual 由三条上游调用路径各自计算一次，生成器
   拒绝出现额外的第四次 residual；训练与评估仍保持上游 PPO 数学、奖励、协方差和
   `log_std` 约定。
2. `FastController` 为每个 simulator environment 保存最近七帧。终止帧先完成当前观测，
   下一次 `act` 前只清空已终止环境；任何 H4 窗口不得跨 episode。
3. 远端先运行 32-iteration/64-env smoke，检查有限性、640 optimizer steps、冻结字段和
   生成 runtime 的 residual-count contract；随后用旧 nonzero adapter checkpoint 做
   128-env deterministic re-eval，不能把零初始化正控当作该审计的替代。
4. 若前两步通过，再以 published state checkpoint warm-start、WAM-v3 step80000、
   `gain=0.05`、seed245、64 env、1000 iterations 跑 corrected bounded adapter。保留
   `model-initial.pt`、每250轮 checkpoint、训练曲线和资源日志；不按 DEV 选模型。

## 判定边界

这是单 seed、短预算 Probe。修复后旧 checkpoint 明显恢复，只能说明旧评估被实现问题混淆；
修复后仍退化，只能把这两个问题从优先根因列表中移除，不能证明 WAM 无效。只有后续 matched
trainable-module、多 seed Validation 才能形成正式支持或反驳。

停止条件：生成器合同失败、非有限值、GPU/磁盘预算越界，或连续八个完整物体段的 critic
EV 近零。任何停止都保留原始产物并标记实现或证据边界。

## 结果

待远端运行结束后补写：source/runtime audit、旧 checkpoint 复评、corrected adapter
训练与固定 TRAIN/DEV 面板、资源与代码 manifest。
