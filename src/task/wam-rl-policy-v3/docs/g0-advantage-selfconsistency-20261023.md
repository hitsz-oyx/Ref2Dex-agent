---
schema: ref2dex.research-report.v1
report_id: P-20261023-g0-advantage-selfconsistency
source_experiment_id: P-20261023-bc-ppo-usefulness
source_run_id: bc-ppo-usefulness-20261023-r1
date: 2026-10-23
task: wam-rl-policy-v3
branch: main
status: INVALID_INSTRUMENTATION
---

# G0 预注册审计：奖励/优势计算自洽性

## brief39 已有数字与本轮边界

开工前已读取 [`ppo-bc-mechanism-diagnostic-20261023.md`](ppo-bc-mechanism-diagnostic-20261023.md) 以及 `outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/` 下的运行 manifest、每回合 `result.json`、`ppo-rollout.npz`、`reward-parts.json` 和 `stages.json`。brief39 已经计算过、且本轮没有重跑 B1/B2/B3 的指标是：

- 全 12D 策略位移均值 **0.01031**、P95 **0.04560**；认证 contact/lift/hold 并集 **15.76%**；event 正奖励 **0.174%**。
- 优势绝对账本 value-TD **85.58%**；brief39 的 bootstrap 账本为 **1.99%**（分母是最终优势绝对量）。
- 10× lr 单步 KL 中位数 **0.02031**、100× **0.12444**。

本轮重建的 raw GAE 均值 **−0.05904524**、标准差 **0.29015872**，与 brief39 的 **−0.059045 / 0.290159** 一致；阶段计数也一致（contact **2,270**、lift **1,466**、hold **1,376**）。本轮另外按预注册要求计算了“置零 bootstrap 两版优势差异 / 两版绝对量之和”，结果为 **0.9876%**。这与 brief39 的 **1.99%** 不矛盾：两者分母不同，而本轮差异的分子 **63.083275** 与 brief39 bootstrap 账本的 **63.08** 一致。策略位移、event 稀疏度、KL 等 brief39 数字在本轮作为既有证据引用，不重复重跑。

本次复核（当前工作树 `HEAD=906d7168`）再次只读同一旧运行；`ppo/manifest.json::git_commit` 仍为 **59fc5f2afcc2d14604aeb2b79ce3d4202296ee98**。实际命令为 `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python tmp/g0_reaudit.py`；脚本只读 outputs，未纳入提交。CPU 扫描 36 个 `ppo/*/ppo-rollout.npz`、合计 14,400 样本，首个文件的实际键集合为 `observations, latents, old_logp, rewards, values`；五个 G0 必需落库字段（`advantage_raw`, `target_return`, `done`, `bootstrap`, `termination_type`）在 **36/36** 个文件缺失。独立复算得到 raw GAE 均值 **−0.0590452381**、std **0.2901587035**，每回合 `target_return_mean` 与 `result.json` 的最大差（按实现的 float32 口径）**2.9802322e−8**，与上面报告数字一致；没有读取或改写任何新 instrumented rollout。

## 预注册结果解读表（原文）

| 观测结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 三项复算全部在容差内（rtol ≤1e-4、atol ≤1e-5）：① `target_return == V_old + advantage_raw` 逐样本成立；② 真实终止/关节安全终止/时间截断三类的 bootstrap mask 与落库一致；③ 同 reward/γ/λ/终止标志重算的 GAE 逐样本一致 | **G0 PASS**（计算链自洽） | 才允许继续讨论奖励设计、正样本覆盖与学习率；G1/G2 的读数才可被当作关于策略/信号的证据 |
| 任一项**无法复现**（超容差、mask 与落库不符、字段缺失） | **G0 FAIL** | 暂停一切"调学习率/加预算"的讨论，**先修计算链**，并写清是 target/mask/GAE 哪一环错了、错在多少样本上 |
| 必要数据缺失或损坏（rollout 文件、mask 或 advantage 未落盘） | **INVALID_INSTRUMENTATION** | 不产生任何执行能力或 PPO 结论；报告缺哪个字段、归属哪个实验版本 |
| 截断终点置零 bootstrap 后，两版优势绝对差占比 ≥50% | critic 校准需要进一步验证（**不得直接判为 bug**，按 GPT 口径） | 把"critic 校准"列为下一项离线检查，而不是去改学习率 |

## 单一裁决

**INVALID_INSTRUMENTATION。** 36 个 PPO rollout、14,400 个 control 的 reward/value 数据可读，且可以重建实现使用的 GAE；但是逐样本 `advantage_raw`、`target_return` 和终止/bootstrap mask 都没有落盘。因此 A 的逐样本落库比较、B 的落库 mask 对照、C 的“复算优势 vs 落库优势”比较无法成立，不能据此给出 G0 PASS 或 FAIL，也不产生执行能力或 PPO 结论。

## 输入、实现口径与实际命令

证据根目录为：

```text
outputs/wam-rl-policy-v3/bc-ppo-usefulness-20261023-r1/
```

主 PPO 运行 manifest 的运行代码提交为 `59fc5f2afcc2d14604aeb2b79ce3d4202296ee98`，状态 `COMPLETED`，配方 `gamma=.96`、`gae_lambda=.95`。本轮审计时仓库 `HEAD` 为 `31436326c4c31aa846008d14b41adac9aebcfac3`。输入 spot-check：

```text
manifest.json sha256 = 0ba9d6b1438d4e7913c5feb6f9cfcdb4a206eaa51e610b300a26e7425f387bcc
ppo/p0000-A1/ppo-rollout.npz sha256 = 155944fcbdfa7c8ae6b6c428cfec615ffcad17c0d450ed19ffcfb69747b488f6
```

实际使用的 CPU 命令为：

```sh
CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python tmp/g0_offline_audit.py
```

`tmp/g0_offline_audit.py` 是本轮临时审计脚本，执行后已删除；脚本只读取上述 outputs 和 task 实现，没有启动训练、采样、物理或 GPU，也没有写入或移动 outputs。实现口径来自运行时冻结的 `ppo/runner.py` 和仓库 `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py`：

```text
val = [V_old[0], ..., V_old[T-1], bootstrap]
for t = T-1 ... 0:
    delta = reward[t] + gamma*val[t+1] - val[t]
    running = delta + gamma*lambda*running
    advantage_raw[t] = running
target_return[t] = V_old[t] + advantage_raw[t]
```

该 runner 没有 per-step `done` 数组；终点是否 bootstrap 通过 `bootstrap` 标量隐式表达。PPO 更新前另做 episode 内标准化，所以 `result.json` 中的 `advantage_std=1.0` 不是 raw advantage 的分布证据。

## A. target_return 与 `V_old + advantage_raw`

落库字段审计结果：

| 需要的字段 | 实际落库 | 结果 |
|---|---|---|
| `V_old` | `ppo-rollout.npz::values` | 有，36 个文件、合计 14,400 样本 |
| raw reward | `ppo-rollout.npz::rewards` | 有 |
| `advantage_raw` | 无 | 缺失 |
| per-sample `target_return` | 无；只有 `result.json::ppo_update.target_return_mean` | 缺失 |

因此按预注册要求的逐样本不一致数、总数和最大偏差**不可判定**，不能把重建数组当作落库数组。作为可复核的代理检查，按实现从 reward/value/bootstrap 重建后，在同一重建数组内部 `target_return - (V_old + advantage_raw)` 为 **0/14,400**，最大绝对偏差为 **0**（这是定义恒等式，不是独立落库验证）；36 个 episode 的重建 `target_return_mean` 与落库均值比较，超 `1e-5` 的为 **0/36**，最大绝对均值差 **2.98e-8**。后者支持数值实现口径一致，但不能补回缺失的逐样本字段。

## B. 三类终点与 bootstrap mask

为便于离线重建，本报告将 `physical_table_penetration` 和成功完成归为“真实终止”，将 `physical_joint_limit` 归为“关节安全终止”，无 abort 且达到 600 控制步的回合及末尾预算缩短回合归为“时间或截断”。统计如下：

| 终点类 | 回合数 | controls | 回合示例/组成 | 落库 bootstrap 数值 | 按协议应有 |
|---|---:|---:|---|---|---|
| 真实终止 | 7 | 2,349 | table penetration 5；success 2 | 7/7 为 0 | 终止置零 |
| 关节安全终止 | 11 | 1,827 | 全部 `physical_joint_limit` | 11/11 为 0 | 终止置零 |
| 时间或截断 | 18 | 10,224 | 17×600 控制时间截断；`p0035-A1` 为 24 控制的预算截断 | 18/18 为非零（含 −0.43703288） | 截断 bootstrap |

从 endpoint 摘要和 `ppo_update.bootstrap` 推断，**没有发现 mask 语义疑点**：真实/关节终止均为零，时间/预算截断均使用非零 bootstrap；`p0035-A1` 的 24-control 预算截断也没有被误当作真实终止。但 `done`/mask 没有逐步落盘，以上是代码路径与终点摘要的一致性检查，不能称为“mask 与落库逐样本一致”。

## C. 独立 GAE 重算与置零 bootstrap 对照

使用每个 rollout 的原始 `rewards`、`values`、端点摘要推断的 bootstrap、`gamma=.96`、`lambda=.95`，按上面的实现独立重算 14,400 个 raw advantage。逐样本与落库 advantage 的比较为：

| 比较 | 最大偏差 | 相关 | 均值/方差 |
|---|---:|---:|---|
| 重算 raw advantage vs 落库 `advantage_raw` | **不可计算：字段未落盘** | 不可计算 | 落库均值/方差不可计算；重算均值 **−0.05904524**、方差 **0.08419204** |
| 重算 target mean vs `result.json` 均值 | 36 回合最大 **2.98e-8** | 不适用 | 36/36 回合均值在 `1e-5` 内 |

重算 raw advantage 分布为：均值 **−0.05904524**，std **0.29015872**，P50 **−0.04258693**，P90 **0.26390052**，P25 **−0.21927930**，P75 **0.11132685**，IQR **0.33060615**，范围 **[−2.016768, 1.331487]**。这与 brief39 的 raw 分布一致，并显示 `advantage_std=1.0` 主要来自 episode 内标准化，而非 raw 信号本身。

将所有时间/截断回合的端点 bootstrap 置零，真实/关节终止保持不变，得到：

```text
sum(abs(A_bootstrap))       = 3174.0230
sum(abs(A_zero_bootstrap))  = 3213.3687
sum(abs(A_bootstrap-A_zero))=   63.083275
max(abs(A_bootstrap-A_zero))=    0.9099039
corr(A_bootstrap,A_zero)    =    0.9927457
63.083275 / (3174.0230 + 3213.3687) = 0.009876219 = 0.9876%
```

因此本轮预注册要求的“两版优势绝对差占两部分绝对量之和”是 **0.9876%**，远低于 50% 的 critic-calibration 触发线。该数字仅是 counterfactual 量化，不修复缺失的落库 mask/advantage，也不改变本报告的 INVALID_INSTRUMENTATION 裁决。

## D. raw advantage 与每初态 episode return

raw advantage 的分布已在 C 列出；各初态按 6 个 PPO 训练 episode 计算 reward return 的总体方差（`ddof=0`）和 IQR 如下：

| 初态 | episodes | return 方差 | return IQR |
|---|---:|---:|---:|
| `s1_cylindersmall_pass_1` | 6 | 0.11585663 | 0.59760764 |
| `s10_alarmclock_pass_1` | 6 | 0.04798865 | 0.38053757 |
| `s10_banana_pass_1` | 6 | 0.07326542 | 0.34028783 |
| `s10_cubesmall_inspect_1` | 6 | 0.09102459 | 0.45849740 |
| `s10_cup_drink_1` | 6 | 1.04163871 | 0.63835332 |
| `s10_cylindermedium_inspect_1` | 6 | 0.61910419 | 0.89835161 |

合计 36 回合的 return 方差为 **0.42835544**、IQR **0.65585149**。因此不能把 `advantage_std=1.0` 解释成原始优势天然稳定；它是标准化后的固定尺度，raw advantage 和 episode return 都保留明显方差。

## E. 正优势的阶段归属

按 `stages.json` 的 `stages[1:n+1]` 与 rollout action 对齐；阶段标志是嵌套认证状态，所以各行不是互斥分区。正优势计数如下：

| 阶段标志 | 认证状态 controls | 正优势 controls | 正优势比例 | 正优势和 |
|---|---:|---:|---:|---:|
| approach | 2,319 | 1,241 | 53.51% | 186.1746 |
| contact | 2,270 | 1,214 | 53.48% | 176.1719 |
| lift | 1,466 | 1,070 | 72.99% | 134.6261 |
| hold | 1,376 | 981 | 71.29% | 121.0324 |

按正优势占比和正优势总量，lift/hold 最集中；approach/contact 也有大量正优势，不能说正优势只出现在最终阶段。对于 11 个 joint abort 回合，排除触发 abort 的最后一个 action 后，前置动作共 **1,816** 个，其中 **822** 个为正优势（**45.26%**），且 **11/11** 个 joint-abort 回合都存在至少一个 abort 前正优势动作。也就是说，joint abort 之前确实“反而”出现正优势；这描述的是当前重建 raw GAE 的信号，不是成功因果证据。

## 卡点与后续最低修复

本轮唯一阻断是 instrumentation：运行只保存了 reward/value 和 episode 级 bootstrap，缺少逐样本 `advantage_raw`、`target_return`、`done`/bootstrap mask。应在后续离线可复核运行中把这三个数组（以及 endpoint 类型/每步 mask）写入 rollout sidecar，再重复 A–C；在此之前不把本轮结果升级为 G0 PASS，也不据此调学习率或补预算。

## 结尾裁决（中文）

三项复算结论是：target 与 `V_old + advantage_raw` 的逐样本落库核对因缺 `target_return/advantage_raw` 不能判定；三类终点的落库 mask 因缺逐步 mask 不能判定，但 episode 级 bootstrap 与终点摘要没有发现语义疑点；GAE 可按实现重建且 episode 均值在 `2.98e-8` 内，但与落库优势的逐样本比较因缺 `advantage_raw` 不能判定。置零截断 bootstrap 的两版优势差异比例为 **0.9876%**。正优势在 approach/contact/lift/hold 均出现，lift/hold 占比最高；11/11 个 joint abort 回合在触发前都有正优势动作（822/1,816）。卡点是逐样本 advantage、target_return 和 done/bootstrap mask 未落盘，所以单一裁决为 **INVALID_INSTRUMENTATION**。
