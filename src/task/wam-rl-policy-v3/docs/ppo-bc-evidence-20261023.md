---
schema: ref2dex.research-report.v1
report_id: P-20261023-ppo-bc-evidence
date: 2026-10-23
task: wam-rl-policy-v3
experiment_id: P-20261023-ppo-bc-evidence
branch: main
status: EVIDENCE_ONLY
---

# 执行器轴：PPO/BC 取证与最小可判定设计

| 结果 | 含义 | 决定 |
|---|---|---|
| 1e5 步内出现改善且不回退 | PPO 本身有用 | 白天排 1e6 卡时 |
| 1e5 步无改善但 KL 已正常 | 预算/覆盖不足，**不是** PPO 无效 | 只扩预算，不改算法 |
| KL 修不好 | 实现问题未排除 | 不许下“PPO 不成立”的结论 |

本轮门槛仍未完全落地：真实部署路径 `ptv3-shared-wam-v6/.../online_runtime.py:49` 仍硬编码 `object_goal_valid=np.ones(32,bool)`。因此不启动 v6 长训练，也不把历史玩具规模的 826 controls/2 updates、0/2 success 当作 “PPO 无效”。

## 1. 首步 KL 的三种含义

证据来自冻结运行 `outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1`，环境为 graspenv Python 3.8.20、Torch 2.4.1+cu121。旧实现快照可用：

```bash
git show 6e60717773cc62c7c08b56d18fa3dff241f9d9b3:src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py
```

| 含义 | 判定 | 证据 |
|---|---|---|
| (a) 自适应 KL 早停逻辑写错 | **确认，是实现 bug** | 旧 `intent_residual.py:51–53` 使用 `(old-logp).mean()`；`old` 是旧策略对 sampled latent 的 log-prob，不是 Normal-to-Normal KL。且 `if epoch and kl>.03` 跳过 epoch 0，所以首个 optimizer step 没有 KL 门。 |
| (b) lr 过大造成首步漂移 | **确认是强贡献因素，不能单独归因** | worker 旧配置 `run_grasp_geometry_worker.py:104` 为 Adam lr=`1e-4`。离线重放同一 p000 batch、同一 BC 起点、无仿真：lr 1e-4 → exact KL .61189、clip fraction .8451；2.5e-5 → .03811/.3982；1e-5 → .00609/.0664；5e-6 → .00152/0。说明步长足以解释首步漂移，但没有把 lr 单变量在线重训，因此不把它写成唯一原因。 |
| (c) target_kl 与策略尺度不匹配 | **确认存在尺度解释，不能替代 (a)** | target=.03 是 12 维 joint KL；旧策略 logstd=-2.5（σ≈.0821），对应每维 target=.0025、均值 L2 约 `sqrt(2σ²·.03)=.0201`。实际 p000/p001 的 mean-shift L2 均值 .0850/.0440，exact joint KL .611887/.165042；因此 .03 对当前 12D latent/σ 是严格门，必须明确 “sum over dimensions” 和 per-dim 统计。它解释阈值敏感性，但不能把错误的 sampled-logp 检查变成正确实现。 |

原始逐回合指标（来自 `ppo/p000-A1/result.json`、`ppo/p001-A1/result.json` 和 `ppo-policy-audit.json`）：

| 回合 | controls | 旧 approx_kl | 精确 Gaussian KL | clip fraction（离线重放） | entropy（更新前→后） | reward sum | 阶段/回报 |
|---|---:|---:|---:|---:|---:|---:|---|
| p000-A1 | 226 | -2.0044e-7 | .611887 | .8451 | -12.972737→-12.972737 | -1.181455 | approach/contact/lift/hold 全 0；time+effort 负项 |
| p001-A1 | 600 | 1.0371e-7 | .165042 | .6800 | -12.972736→-12.973036 | -.845830 | approach/contact/lift/hold 全 0；time+effort 负项 |

两回合的 `onpolicy_logp_max_error` 为 1.81e-5/1.91e-5，GAE 重放误差 3.73e-9；所以问题不是采样 log-prob 回放漂移。旧实现把首步 KL 误报成近零，而精确审计明确超出 .03（20.4×/5.5×）。熵几乎不变，说明本次漂移主要来自 mean，而非探索方差膨胀。reward-parts 中两回合 shaping/event/sparse 均为 0，不能从这两条回报序列推断任务收益。

已在 `intent_residual.py:41–114` 修复：保存 pre-step policy/optimizer，首步也计算 `torch.distributions.kl_divergence(old_dist,new_dist).sum(-1).mean()`；超 target 时回滚并以 1/4 lr 重试，四次仍超则拒绝该 epoch；每轮记录 exact/approx KL、clip fraction、entropy、lr、loss、return/reward 摘要。worker 改为 Adam lr=`1e-5`、最多 2 epochs，并显式传 `target_kl=.03`。本轮只做代码与离线数值验证，没有启动规模训练。

## 2. 吞吐、预算和写盘

已核实基准为约 15 environment controls/s：52,839 controls 约 1 h，因此：

| 预算 | 每个 arm 的墙钟估算（15/s） | BC-only + BC+PPO 配对总量 |
|---:|---:|---:|
| 1e5 controls | 1.85 h | 2e5 controls，约 3.70 h（若两 arm 同时跑则墙钟受最慢 arm 约 1.85 h） |
| 1e6 controls | 18.52 h | 2e6 controls，约 37.04 h（同时跑的理想上界约 18.52 h） |
| 1e7 controls | 185.19 h（7.72 d） | 2e7 controls，约 15.4 d |

预注册决策的最低可判定 probe 是 **每个 arm 1e5 controls**，并在同一 3+ 初态面板上比较成功率或关节限位；只有 KL 正常而 1e5 无改善，才把同一配方扩到每 arm 1e6。今晚不排 1e6。

4 卡线性加速只能作为上界，不能当已证事实。若把已测 15/s 当作当前三卡 aggregate，理想的 4/3 扩展是 20/s、1e5 为 1.39 h、1e6 为 13.89 h/arm。已有三 worker 运行的实测并不稳定：v11 三卡 manifest 的控制/最长 wall 给出 52,839/3,904=13.53/s；v9 三卡给出 101,570/3,210=31.64/s。差异来自对象、物理后端、reset/日志和负载，且没有四卡同步 PPO 的测量。因此本轮预算按 15/s 保守记账；白天 preflight 只测 10–20 min 的四卡 aggregate controls/s 和 step-time P50/P95，再决定是否按 4/3 近似。

写盘按已存在 trace 的实测量估算：v11 388 MiB/52,839 controls≈7.1 KiB/control；当前 residual pilot 43 MiB，历史 `outputs` 约 571 GiB，`/home2` 可用约 140–141 GiB，历史产物不删除。采用 10 KiB/control 的保守上限：

- 1e5/arm：约 1.0 GiB/arm，配对约 2.0 GiB；保留 64 MiB checkpoint、manifest 和日志后仍远低于余量。
- 1e6/arm：约 10 GiB/arm，配对约 20 GiB；若每 1e5 保存一次 61 MiB 级模型，再加约 1–2 GiB。
- 1e7/arm：约 100 GiB/arm，配对约 200 GiB，超过当前余量，不能预注册为本地无界输出。任何新 run 必须设置 output cap、checkpoint 间隔和低于 20 GiB 的停止条件。

## 3. 最小可判定实验设计（只写设计）

- **比较**：同初态、同 controls 预算、同观测/动作空间/相位机、同 BC 模型和 frozen R/WAM；两 arm 仅为 `BC-only` 与 `BC+残差PPO`，不换模型、不加新 baseline。
- **预算顺序**：先每 arm 1e5 controls；3+ 个预注册初态在两 arm 中成对出现。若 KL 仍超 target，实验无效，回到实现修复；若 KL 正常但无改善，只扩每 arm 1e6，不改算法。
- **判据**：成功率上升或关节限位下降，且至少 3 个初态改善、没有一个初态回退；同时报告 exact joint KL、per-dim KL、clip fraction、entropy、episode return 和 reward 分项。
- **边界**：本轮 0/2 只是玩具 pilot 的工程现象；不使用它下任何 “PPO 无效” 结论。

## 4. 修复方案与 preflight（不放规模）

修复已经落到代码，但不启动规模训练。下一次运行前先做 10–20 分钟 preflight：

1. 在 GPU 4/5/6/7 中只选空闲卡，固定同一 BC checkpoint、同一 3+ 初态和同一 registration；先用 10–20 min 估计 controls/s、显存峰值和每回合 P50/P95。
2. 用新日志检查每个 update 的 `exact_kl <= .03`、`clip_fraction`、`entropy`、`return_mean/sum`、reward 分项和 `accepted`；任何 non-finite、KL 拒绝后仍改变权重、或 log-prob replay >1e-4 都停止。
3. 若首步仍超 target，优先确认 old/new distribution 的 12D sum 单位和 lr retry；不要调大 target 来掩盖实现问题。
4. 预期修复配方：Adam lr=1e-5、最多 2 epochs、exact joint target_kl=.03、四次 1/4 lr rollback/retry、保持 action ratio clip=.2 和现有 gamma=.96/GAE=.95。通过 preflight 才允许排 1e5 probe。

复现/检查命令（均为离线或短 preflight，不是长训）：

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
PYTHONPATH=src/task/wam-rl-policy-v3/src $PY -m py_compile   src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py   src/task/wam-rl-policy-v3/tools/run/run_grasp_geometry_worker.py
PYTHONPATH=src/task/wam-rl-policy-v3/src $PY - <<'PY'
import numpy as np, pathlib, torch
from wam_rl_policy_v3.intent_residual import IntentResidualPolicy, ppo_update
root=pathlib.Path("outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1")
z=np.load(root/"ppo/p000-A1/ppo-rollout.npz")
p=IntentResidualPolicy()
p.load_state_dict(torch.load(root/"bc/bc.pt", map_location="cpu", weights_only=False)["model"])
r=ppo_update(p, torch.optim.Adam(p.parameters(), lr=1e-4),
             z["observations"], z["latents"], z["old_logp"],
             z["rewards"], z["values"], target_kl=.03, epochs=2)
print(r)
assert r["final_exact_kl"] <= .03 + 1e-6
PY
pytest -q src/task/wam-rl-policy-v3/tests/test_distribution_kl.py
```

该命令只读取已有 rollout/checkpoint，验证修复会回滚到 target 内；它不启动 v6、BC 或 PPO 长训。
