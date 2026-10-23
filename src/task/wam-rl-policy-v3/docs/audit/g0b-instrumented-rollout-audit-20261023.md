# G0b：仪器化 rollout 逐样本自洽审计（2026-10-23）

本轮新增：仅持久化补丁、同一旧批次的补丁前后等价性核对，以及新目录中最多每臂 1,500 controls 的 BC/PPO 最小物理复跑与逐样本 G0 重算。引用而不重跑：brief42 的旧 36 回合/14,400 samples 及 INVALID_INSTRUMENTATION；brief39 的 G1/G2/G3 和 B2/B3。旧读数不作为本轮新数据证据。

## 结果解读表（运行前预注册；不得事后改判）

| 观测结果 | 含义 | 会改变哪个决定 |
|---|---|---|
| 补丁后新跑出的 rollout 落全三类字段；据此逐样本复算 ①`target_return == V_old + advantage_raw`、②三类终点 bootstrap mask、③同 γ/λ/终止标志重算 GAE **全部**在容差内（rtol ≤1e-4、atol ≤1e-5） | **G0b PASS**（计算链与落盘自洽） | 才允许回到"奖励设计 / 正样本覆盖 / 学习率 / 预算"讨论；brief39 的 G1/G2/G3 读数才可被当作信号证据 |
| 补丁后仍缺字段、或落库字段语义与实现不一致（例如落的是**标准化后**的 advantage、mask 只有 episode 级、target 用了 bootstrap 后的 return） | **G0b FAIL（工具链）** | 停：先修落盘与语义，不碰 lr / 预算 / 奖励形状 |
| 三项复算超容差；或**真终止/关节安全终止被 bootstrap**、**时间截断未 bootstrap** | **G0b FAIL（计算链错）** | 写出错在哪一环、多少样本；列为最高优先修复单点，执行器线其余工作暂停 |
| 三项通过、但截断置零 bootstrap 后两版优势绝对差占比 ≥50% | critic 校准需进一步验证（**不是** bug） | 下一项离线检查 = critic 校准，而不是去改学习率 |
| 最小复跑出现 OOM / 非有限 loss / loader failure / `/home2` 可用 <20GiB | 故障门 | 立即停、**不自动重跑**、保留证据、不删历史产物 |

## 冻结规模与边界（计算前登记）

Decision：只区分 G0 计算/记录错误与自洽，不推断 PPO 学习效果。最便宜方法：一批旧样本做等价性验证，新物理 rollout 仅补齐旧数据不存在的逐样本字段。

BC/PPO 各最多 1,500 controls，总计最多 3,000；PPO 更新最多原实验 72。沿用旧注册六初态的任务顺序、模型 seed 293、native seeds 294–299、原始 checkpoint、γ=0.96、λ=0.95、lr=1e-5、两 epochs/episode、KL≤0.03、600 controls 常规回合时域。预算余数造成的末回合须作为 truncation 单列；不延长预算以凑终点类别。只用 GPU7，已用显存>2GiB 时等待并每 60 秒重试。不得触及 GPU0–6 或 tau-checkpoint-ab 中途产物。

新产物根：`outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/`。预估总写盘<512MiB，硬限额1GiB；磁盘<20GiB、OOM、loader failure 或非有限量立即停止且不自动重跑。源码只增加 PPO 落盘所需记录，不改计算语义。覆盖自评必须依据实际终点；任何未观测类别明确标未覆盖。

状态：预注册先落盘，随后提交补丁 `64f04603` 并启动一次最小复跑；启动故障后按门禁停止，未自动重试。最终裁决见下。

## 补丁与运行前等价性证据

`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py::ppo_update` 增加可选 `audit` 接收器；在原始 GAE 算完、标准化之前复制 `advantage_raw`、实际进入 value loss 的 `target_return`、`V_old`、reward、next value、γ/λ 和末状态 bootstrap 值。原计算表达式、PPO 损失、Adam、KL gate、两 epochs 均未修改。runner 仅把这些副本与每样本 done/bootstrap/终点标签及已存在的实测阶段一起写入 `ppo-rollout.npz`。

`done` 是回合边界标志（所有回合最后一个样本为 True）；`bootstrap` 是一步 TD 的 next-value 有效标志（中间样本为 True，真正终止/关节安全终止末样本为 False，时间/预算截断末样本为 True）。`termination_type` 中间样本为 continuing，末样本为三类之一；GAE trace 在任何 done 处停止递推，但截断 TD 仍使用末状态 value。这些字段描述原 runner 的整回合计算，不反过来控制更新。

等价性：使用旧 PPO `p0000-A1/ppo-rollout.npz` 的 297 样本、同 BC checkpoint、seed 293，在 CPU 上分别调用原冻结函数与补丁函数各一次（各 2 optimizer updates）。6/6 比较逐位相同：raw advantage、target、标准化优势、全部更新指标、模型 state_dict、Adam state_dict；最大 raw advantage 差=0、最大模型参数差=0。两 epoch loss 两边都是 `[0.08426089584827423, 0.06399012356996536]`。这是同 CPU 批次的严格等价，不声称 CPU 与旧 GPU 浮点执行逐位相同。原 GPU 数据在 CPU 的行为 logprob 重建最大差 2.288818359375e-5，低于原 gate 1e-4。

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/check_equivalence.py
```

预期：`equivalence.json::status=PASS`，所有 checks=true，两侧 updates=2，raw/参数最大差=0。该核对不新增环境交互。

运行环境：policy Python3.8.20 / torch2.4.1+cu121 / NumPy1.24.4，native 使用原 Python3.8.10 / torch2.0.1+cu118 环境。冻结旧 PPO runtime，唯一算法包差异为上述落盘副本；原 R、v6、BC residual 权重路径与数据源均保持；本轮在 WAM 依赖导入时失败，未完成模型加载与物理启动。输出根 scratch 替代 runner 原来的仓库 tmp，以符合本轮写入边界；路径超过 AF_UNIX 的 108-byte 限制时，输出目录内 `short_unix.py` 只在 socket bind/connect 系统调用处使用同一文件的 `/proc/self/fd` 短别名，137-byte 路径的原样 payload 往返已通过。native wrapper 仅安装此传输适配，不修改物理或模型代码。实际子进程命令另存 `*-native-commands.jsonl`。

```sh
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -u outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/supervise.py
```

预期：GPU7 按顺序 BC→PPO，各≤1,500 controls，无额外 assessment；故障立即 STOPPED_NO_RETRY。注册完整保留旧六初态顺序与 seeds，只缩小交互/墙时/写盘上限。

补丁 SHA256（完整 diff 在新输出根 `instrumentation.patch`）：

```json
{
  "src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py": "51a4d9176d4fa1d407e58a6f53df3724b83e91e20cefb8192fa25a91be4344d4",
  "src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py": "cf8bcfe757f5cea42bb189f59d12e24927729ed27393ba26c530c5101a24b858",
  "patch_sha256": "fe3f51f545c243a26650b0b00e3ba4e5505984043d66174245ecd001d7057c83",
  "old_runner_sha256": "6d966f48ca6cced7932f79cda444f1eea330ce89ac46cf80dc1218843364895a",
  "old_residual_sha256": "d0f5efd2c6e0360b46edf7ddee3733b0e16bec84daca156b1713e2de74aef0df"
}
```

## 实际复跑：故障停止，无新 rollout

**单一裁决：G0b INVALID。** 新数据不存在，无法判定 GAE 计算链 PASS 或 FAIL；这次失败属于我新增的运行包装/产物准备错误，不是 PPO 能力、奖励或学习率的负证据。源码的逐样本落盘分支尚未被新物理 rollout 执行验收，不能用 CPU 等价性 PASS 代替 G0b PASS。

实际启动时间 2026-10-23 08:12:42 +0800，08:13:12 监督器记录 STOPPED_NO_RETRY。GPU7 进入和退出均 `7, 2, 0`（MiB / utilization %）。没有使用其他 GPU，没有启动 native 物理进程，没有重试，没有 assessment rollout。

| 项目 | 预注册上限 | 本轮实际 |
|---|---:|---:|
| BC controls | 1,500 | **0** |
| PPO controls | 1,500 | **0** |
| PPO 物理采样后的 learning_updates | 72 | **0** |
| native process.json / trajectory.npz / ppo-rollout.npz | 按需要 | **0 / 0 / 0** |
| CPU 旧批次等价性核对 | 1 批、两版各一次 | **297 样本；6 checks passed / 0 failed；两版各 2 updates** |

两处独立故障均保留：

1. **Loader 启动失败。** `launch_arm.py:15::mkdtemp` 使用 `*args` 转发，并额外传入 `dir=...`；`wandb` 导入中 `TemporaryDirectory` 通过 `mkdtemp(suffix, prefix, dir)` 传入第三个位置参数，导致 `TypeError: mkdtemp() got multiple values for argument 'dir'`。调用链在 `run_bc_ppo_probe.py:124 → OnlineV6Runtime.__init__ → model.py → PointWorld/timm → wandb`。此时尚无物理进程、轨迹或 PPO 更新。`bc-worker.log` 留有完整堆栈。
2. **产物上限故障。** 旧 frozen runtime 的 `third_party` 是指向仓库 `third_party` 的软链接。预估只数了旧树的普通文件，约 2.29MB；`shutil.copytree` 默认展开软链接，实际复制了依赖仓库/历史 checkpoint 等约 1.57GB，超过预注册 1GiB。监督器首轮检查因此报 `RuntimeError('storage/wall/update fault')`。这是我的写盘估计与复制方式错误，不是磁盘<20GiB。停止时仍可用 127.70GiB；保留产物，未删除任何历史输出，也未擅自删除本轮证据。

监督器摘要中的首条错误是 output cap，worker 日志另有 loader TypeError；不能只看 supervisor-result.json 而遗漏后者。由于用户硬规定“最小复跑出现 … loader failure … 立即停、**不自动重跑**”，本轮未修包装器后再次启动，也没有改预算、γ/λ、奖励或模型来绕过故障。

## 新数据 G0 三项及 mask 覆盖（不以旧数据补数）

| 逐样本比较 | 不一致样本 / 总数 | 最大绝对偏差 | 结论 |
|---|---|---|---|
| `target_return == V_old + advantage_raw` | **未取证 / 0** | **N/A** | 无新 rollout |
| 三类 endpoint 的 done/bootstrap mask | **未取证 / 0** | 不适用 | 无新 rollout |
| 同 reward/γ/λ/done/bootstrap 独立重算 GAE | **未取证 / 0** | **N/A** | 无新 rollout |

| 终点类 | 新回合数 | controls | mask 不一致数 | 覆盖 |
|---|---:|---:|---|---|
| real_termination | **0** | **0** | N/A | 未覆盖 |
| joint_safety_termination | **0** | **0** | N/A | 未覆盖 |
| truncation（含预算截断） | **0** | **0** | N/A | 未覆盖 |

**最小规模自评：该类预算截断 mask 在本轮规模下未被覆盖。** 1,500 非 600 整数倍，并不保证出现预算截断：途中物理中止可能改变余数，且本次在首个控制前已失败。不能由注册预算的余数冒充观测到的边界。本轮三类终点均未覆盖。

新数据两版优势绝对差占比 `sum|A_boot−A_zero| / (sum|A_boot|+sum|A_zero|)`：**N/A（0 样本）**，不触发或否定 50% 的 critic 条件。新 raw advantage 均值 / std / P50 / P90 / IQR：**全部 N/A**。正优势按 approach/contact/lift/hold 的数量与比例：**全部未取证（总样本 0）**。没有把 brief42 的旧 0.9876% 或任何旧分布填入新数据栏。

## 产物、diff 与复核命令

根目录：`outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/`，outputs 不入库。`source-sha256.json` 与 `instrumentation.patch` 固定源码/补丁；`equivalence.json` 固定旧批次等价性；`bc-worker.log`、`supervisor.jsonl`、`supervisor-result.json`、`audit-result.json` 保留故障与零采样证据。上面的 supervise 命令是**已执行的历史命令**，不是授权重试命令，现有子目录也不满足 runner 的 fresh-output 条件。

可复核的最小 diff（源码提交 `64f04603`）：

```sh
git show 64f04603 -- src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py
```

精确位置：`intent_residual.py:41::ppo_update` 可选接收器，`:63` 复制标准化前 witness；`run_bc_ppo_probe.py:282` 传入接收器，`:294` 构造逐样本边界，`:300` 对齐 action 后 stages，`:302` 一并落盘。旧 GAE 仍在 `intent_residual.py:57`，旧 bootstrap 分支仍在 runner `:273`；本轮均未改表达式。

只读复核故障与零样本（预期 audit-result verdict INVALID，counts 全 0）：

```sh
cat outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/audit-result.json
cat outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/supervisor-result.json
tail -n 10 outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/bc-worker.log
nvidia-smi -i 7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

## 卡点与下一轮最小补救（本轮不执行重试）

下次获得复跑指令后，先仅修运行包装器：显式采用 `mkdtemp(suffix=None, prefix=None, dir=None)` 签名，内部只传一次受限 dir；先以 `TemporaryDirectory` 的同一位置参数调用做无 GPU 启动检查。冻结树必须 `copytree(..., symlinks=True)` 保留原依赖链接，并在任何进程启动前核对实体字节数≤1GiB。当前错误包装器与展开的树保留作证据，不覆盖；新 attempt 使用本轮根内新子目录且不再物化 third_party。两项补救不触及 reward/GAE/模型/执行器。通过后才能在 GPU7 按原上限重新申请最小 rollout；不能把本次 0 samples 算作 G0 验收完成。

Git：第一次提交补丁与预注册后，`git pull --rebase origin main` 因他人未暂存修改被拒绝；未 stash/未改他人文件，普通 push 成功。只提交本轮两个持久化源码文件与本报告。最终报告提交同样显式路径推送，并在答复中给出 HEAD / origin/main / ls-remote 的终态核对。

终态产物快照：**1572847334 bytes**（9259 个实体文件；不重复累计 `bc/runtime-code` 目录软链接目标），约 1.464828GiB。磁盘剩余 127.68GiB。该超限已如实列为本轮故障；未删除产物。


## brief45 增补 v2：R1–R4 预注册（新一轮，实施前写入）

本节新增数据与 08:12 的失败 attempt 严格分开，失败 attempt 保持 INVALID。当前用户增补授权新的最小复跑；不自动恢复旧进程。保留顶端原始解读表，并按本节要求验收双层门：真实短跑落盘 + 确定性边界注入。

- **R1：** 逐样本扩展 episode_id/env_id/timestep/sample_index、reward/value_t/bootstrap_value、advantage_raw/target_return、terminated/truncated/endpoint_type、bootstrap_mask/gae_continuation_mask。`b_t=1−terminated_t`；`c_t=1−(terminated_t or truncated_t)`，片段末端（含 budget cutoff）c=0，time limit/cutoff 的 b=1。bootstrap_value 是本样本 TD 使用的、同一回合 next-state value；另存末状态标量与来源状态身份。独立逐样本重算 δ=r+γbV_next−V、A=δ+γλcA_next、target=V+A，rtol=1e-4/atol=1e-5。
- **R2：** 先抽出原 GAE 表达式供 PPO 与无训练边界案例共同调用，保持旧浮点运算顺序；同一生产保存函数落盘。真实终止、time-limit、rollout/budget cutoff 各≥2例；真终止包含物理失败。构造不同的 final-state value 与 reset-state value，逐例确认无 reset 串入；再故意注入 terminal bootstrap、truncation 清零、跨回合递推、误用 reset value、保存 normalized advantage 等错误验证检查器可检出。此层0环境controls、0训练updates。
- **R3：** 原 14,400 controls 永久 INVALID_INSTRUMENTATION，绝不追认。只把计算公式/优化器设置/相关代码路径不变称为同 recipe；缩短预算不是原训练的等价结果。短跑 PASS 仅认证所验路径，不认证 PPO 学习有效；只有发现长序列/buffer rollover 依赖异常才申请针对性长复跑。
- **R4：** 新 attempt 输出到原授权根内 `v2/`。真实物理仅旧注册前2初态，各臂各一次（2初态×BC/PPO，常规600 controls时域，每臂至多1,200且硬上限1,500；learning_updates≤72）。不凑齐随机终点类别、不额外rollout。确定性边界测试补齐真实短跑未观测类别，并分别报覆盖，绝不混成物理样本。

先修上一轮包装器的位置参数签名；新 frozen runtime 保留软链接，实体字节启动前核算。GPU7占用>2GiB等60秒，不换卡。上轮1.57GB证据保留，**新v2子目录**单独≤512MiB预计/1GiB硬限额，不把已冻结失败证据误计入新attempt上限。新attempt遇OOM/非有限/loader failure/可用磁盘<20GiB即停不自动重试。继续使用 long-running-tasks 操作规范（单次阻塞≤60秒遵循上级限制）。

状态：v2 预注册先落盘；源码、CPU 等价性、生产路径边界注入和启动预检已完成，v2物理采样尚未启动。

### v2 运行前验收（新增，非 brief39/42 复算）

原始整片段 GAE 的表达式与浮点运算顺序移入 `gae_targets`，PPO 仍调用原公式；`final_state_value` 只读取当前片段末、reset 前的 observation；`save_ppo_rollout` 同时保存两种 mask 和身份字段。真实 runner 和确定性边界都调用这三个生产函数，没有另造仅供测试的 GAE。

CPU 同一旧批次再次核对：297 samples，6 checks passed / 0 failed，原版与新版两 epochs 的 loss、raw/normalized advantage、targets、模型及 Adam 状态逐位相同，两版各2次更新。此为本轮额外的一个批次等价性检查，不计入物理 learning_updates；不追认旧训练数据。

确定性边界：**6 cases / 24 samples / 0 controls / 0 training updates**，三类各2。独立双mask重算不一致 **0/24**，最大 GAE 偏差 **5.8654384682554905e-8**；target恒等式偏差0。六片段拼接后仍0/24不一致；故意开启第一个边界的跨回合递推产生4/24超差、最大0.5550143867，证明 c 的检查并非只靠片段末 tail=0 的恒等式。8个落盘错误突变全部检出：terminal bootstrap、截断清零、跨回合c、用reset价值且同步改A/target伪装自洽、normalized替代raw、漏reward、错一帧value、缺独立mask。

| 确定性案例 | sample_index | endpoint_type | 末步 b / c | final-state value / reset-state sentinel | 结果 |
|---|---|---|---|---|---|
| true-success | 0–3 | real_termination | 0 / 0 | 0 / 17 | PASS |
| physical-joint | 4–7 | joint_safety_termination | 0 / 0 | 0 / −19 | PASS |
| time-limit-positive | 8–11 | time_limit_truncation | 1 / 0 | .7 / 101 | PASS |
| time-limit-negative | 12–15 | time_limit_truncation | 1 / 0 | −.4 / 102 | PASS |
| budget-cutoff-positive | 16–19 | rollout_cutoff | 1 / 0 | .9 / 201 | PASS |
| rollout-cutoff-negative | 20–23 | rollout_cutoff | 1 / 0 | −.8 / 202 | PASS |

每例前三步 b=c=1；真实终止不调用critic，其他四例只调用一次并记录输入，必须为final observation且与reset sentinel不同。真实控制器不存在本步自动reset分支，实际短跑仍需以末状态tick/episode和冻结critic独立核对，测试不能代替该证据。

包装器预检 **4 passed / 0 failed**：实际 mkdtemp 位置参数、wandb同口径TemporaryDirectory、长UNIX路径原样payload、启动前空间/字节门。新v2实体大小2,374,605 bytes（预检时），保留third_party软链接，不重复上轮的展开复制。GPU7启动前2MiB，其他卡未用。

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/check_equivalence.py
# check_boundaries.py 为 fresh-output 测试，boundaries 已存在时会显式拒绝覆盖；已有结果可只读复核：
cat outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/boundaries-result.json
# 本轮实际短跑启动命令（后续故障不得自动重试）：
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -u outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/supervise.py
```
