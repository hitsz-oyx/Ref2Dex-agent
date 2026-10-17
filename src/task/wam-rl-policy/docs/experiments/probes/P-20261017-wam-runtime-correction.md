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
status: UNCLEAR
run_id: corrected-wam-{smoke,eval,train,eval-control}-remote-20261017-r{1,2,6}
---

# 修正 WAM adapter 的执行一致性与逐环境 history

Result: 已完成，阶段判断为 `UNCLEAR`。该 Probe 只判断两个已定位的 runtime 问题是否足以改变旧 adapter 负结果
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

## 结果（2026-10-17）

远端成功运行使用 `graspenv` 的 Torch 2.0.1/cu118 与 transforms3d，叠加
`dexplore-4090` 的 gym site-packages；Isaac Gym Python 为
`/home/wbcd/workspace/dex/third_party/isaacgym/python`，`LD_LIBRARY_PATH` 指向
`graspenv/lib`，ninja 使用 `graspenv/bin/ninja`。后续 runner 已将 `TORCH_EXTENSIONS_DIR`
隔离到每个 run，避免共享 JIT cache 的 UID 权限冲突。r1--r5 的失败只属于环境准备失败，
没有进入物理训练或评估，保留在远端输出中。

修正后的 64-env/32-iteration smoke（r6，代码 `10487990`）完成 640 optimizer steps，
`learning-audit=PASS`，adapter 参数变化且 native actor/critic/log_std 保持冻结，末值
critic EV 为 0.871。旧有界 adapter checkpoint 的 128-env TRAIN 重评为
`58/128 success, 40 held60+lift10cm, 63 lifted`；旧 runtime 对同一 checkpoint 的记录是
`65/128, 37, 70`。由于本次同时修正 residual 路径和 history，差异不能分解成单一 bug 的
因果效果。

修正后的 `gain=0.05`、64-env、1000-iteration 训练（代码 `10487990`）完成 512,000
environment steps、20,000 optimizer steps，审计通过，末值 EV=0.918，无 invalid history。
其 `model_1000.pt` 在同 seed249、128-env TRAIN 上为 `67/128 success, 43 held, 70 lifted`。
修正 published zero-init control（代码 `d2533f4d`）为 `65/128, 39, 71`，所以本轮差值为
`+2 success, +4 held, -1 lifted`；这是单 seed/单 TRAIN 面板的弱正向读数，Probe 保持
`UNCLEAR`，不升级为 WAM 有效性结论。

主要证据目录（远端固定 repo
`/mnt/ugreen_nas/storage/oyx_storage/robot-finetune-portable-20261016-r1/repo/outputs/wam-rl-policy/`）为：

- `corrected-wam-runtime-smoke-remote-20261017-r6`
- `corrected-wam-runtime-old-eval-remote-20261017-r1`
- `corrected-wam-runtime-train-remote-20261017-r1`
- `corrected-wam-runtime-eval-remote-20261017-r2`
- `corrected-wam-runtime-published-control-remote-20261017-r1`

下一步不追加本配方长训；若继续，应做相同可训练模块的 native matched control、独立 DEV
面板和多 seed Validation，再判断这点恢复是否可重复。
