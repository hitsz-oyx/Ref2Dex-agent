# G0b：仪器化 rollout 逐样本自洽审计（2026-10-23）

**当前单一裁决：G0b INVALID（最新物理边界覆盖不足）。** v2的计算/软件边界检查已通过，v3新增1,395样本也未发现计算错误；但可逐样本核验的真实关节终止仍0例，未达到每类至少2例，不能把较窄范围的PASS升级为本次扩展验收PASS。此前v2的807个PPO样本与24个软件边界样本通过计算/记录自洽验收；不表示PPO学习有效。v2阶段BC/PPO controls为643/807，更新0/4；v3新增BC/PPO controls为1299/96，更新0/4。旧14,400 controls永久 INVALID_INSTRUMENTATION；08:12失败attempt也保持INVALID，均不追认。

本轮新增（brief45及追加）：仅持久化补丁、同一旧批次的补丁前后等价性核对、v2最小物理复跑与v3六个物理边界选点；累计BC 1,942/PPO 903 controls，各低于追加允许的3,000/臂。引用而不重跑：brief42的旧36回合/14,400 samples及INVALID_INSTRUMENTATION，brief39的G1/G2/G3和B2/B3；旧读数不冒充本轮新数据。v1失败、v2计算通过、v3物理覆盖不足均分别保留。

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

## 历史 v1：补丁与运行前等价性证据

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

## 历史 v1（08:12）：故障停止，无新 rollout

**历史 v1 attempt 裁决：G0b INVALID（保留，不追认）。** 新数据不存在，无法判定 GAE 计算链 PASS 或 FAIL；这次失败属于我新增的运行包装/产物准备错误，不是 PPO 能力、奖励或学习率的负证据。源码的逐样本落盘分支尚未被新物理 rollout 执行验收，不能用 CPU 等价性 PASS 代替 G0b PASS。

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

## 历史 v1：新数据 G0 三项及 mask 覆盖（不以旧数据补数）

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

状态：v2 预注册先落盘，随后通过源码/CPU/边界/启动预检并完成2初态×2臂；实际数字与裁决见下。

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


## R1 落地：字段、定义式、生产入口与新数据核对

仅记录代码允许的显式单环境、每次一个回合片段。生产实现位置：

- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py:41::gae_targets`：原 GAE 表达式与运算顺序原样抽取；`:55::final_state_value`：只查询 reset 前末 observation；`:63::save_ppo_rollout`：实际生产保存函数；`:102::ppo_update` 使用同一 GAE 结果更新，`:107` 额外复制实际进入损失的 target tensor。
- `src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py:273`：构造末状态输入；`:281` 调用 bootstrap 选择；`:282` 保存其来源 tick/native episode；`:287` 保存更新前 critic；`:296` 分类终点；`:297` 调用同一保存函数。

新增必需字段全部为每样本长度 n：`episode_id/env_id/timestep/sample_index`、`reward/value_t/bootstrap_value`、`advantage_raw/target_return`、`terminated/truncated/endpoint_type`、`bootstrap_mask/gae_continuation_mask`。本轮2个PPO文件共807样本，全部15个字段逐样本形状、身份、次序和 mask 验证通过；保存总字段数35。`bootstrap_value[t]` 明确定义为该 transition 使用的 `V_old(s_{t+1})`（真正终止时原程序使用0），不是整个片段共享的一个标量。另存 `bootstrap_terminal_value` 标量以保留原调用实参证据。

```text
b_t = 1 - terminated_t
c_t = 1 - (terminated_t OR truncated_t)
delta_t = reward_t + gamma * b_t * bootstrap_value_t - value_t
A_t = delta_t + gamma * lambda * c_t * A_{t+1}
target_t = value_t + A_t
```

中间样本 b=c=1；真正终止（含关节安全、桌面物理失败、实际成功）末步 b=c=0；time-limit 与 rollout/budget cutoff 末步 b=1,c=0。最后一个 A_next=0。现有生产 GAE 逐片段调用，末端 tail=0 已隐式停止递推；本补丁只把该语义显式记录，未把 mask 作为新算法塞入旧公式。timestep 是 action 的前状态tick（0..n−1），阶段标签是该action执行后的实测状态（stages[1:n+1]）；两者时间口径明确分开。sample_index 在该arm累计递增0..806，env_id=0，episode_id包括arm/回合名。

| 新数据逐样本检查 | 不一致 / 总数 | 最大绝对偏差 |
|---|---:|---:|
| target_return == value_t + advantage_raw | **0/807** | **0.0** |
| 同γ=.96、λ=.95、b/c独立重算GAE | **0/807** | **5.74435918832e-08** |
| bootstrap_mask / gae_continuation_mask / terminated / truncated | **各0/807** | 布尔逐位一致 |

独立检查器 `v2/audit_utils.py` 不导入生产 GAE，按上述双mask公式用float64逐步重算，与实际保存数组按 rtol=1e-4/atol=1e-5 比较；还对 γ/λ 与注册值核对，不能仅靠落库参数的内部自洽。old critic snapshot 在807个原observation上的CPU前向与GPU落库 value 全部在容差内，最大差 **2.45869159698e-07**。

### 新物理样本逐回合台账（不是旧14,400的重评分）

| 回合 / 初态 | native seed | controls | endpoint | 末步 b/c | bootstrap | GAE最大偏差 |
|---|---:|---:|---|---|---:|---:|
| p0000-A1 / s1_cylindersmall_pass_1 | 294 | 600 | time_limit_truncation | 1/0 | -0.0234384164214 | 2.91904190819e-08 |
| p0001-A1 / s10_alarmclock_pass_1 | 295 | 207 | real_termination | 0/0 | 0 | 5.74435918832e-08 |

| 终点类 | 真实新PPO回合数 | controls | endpoint / per-sample mask 错误 | 覆盖解释 |
|---|---:|---:|---|---|
| real_termination | 1 | 207 | 0/0 | 已实测 |
| joint_safety_termination | 0 | 0 | N/A（零样本） | 真实短跑未覆盖；另见确定性边界 |
| time_limit_truncation | 1 | 600 | 0/0 | 已实测 |
| rollout_cutoff | 0 | 0 | N/A（零样本） | 真实短跑未覆盖；另见确定性边界 |

此处 real_termination 是207步的 `physical_table_penetration`，不宣称任务成功。BC另有643 controls，但不进行PPO更新、不产生GAE张量，不能混入807个PPO样本分母。

## R2 落地：三类边界、reset前价值与故意错误

上方六案例台账是本轮真实调用生产 `gae_targets/final_state_value/save_ppo_rollout` 得到，true-success 只是测试案例ID、不是仿真成功轨迹；奖励与values是明确构造的软件fixture。每例 rewards=`[.1,−.02,.3,−1]`（真正终止）或`[.1,−.02,.3,.04]`（两种截断），values=`[.2,−.1,.4,.15]`。三大类：真正终止2例（普通终止与关节物理失败各1）、time limit 2例、rollout/budget cutoff 2例，共24样本。终止类的critic调用数0，截断类各1；有意设置完全不同的reset-state value（17/−19/101/102/201/202）作反例。ValueProbe只是无训练的软件测试替身，不是新增物理基线。

**6 cases passed / 0 failed，8 mutants killed / 0 survived**。复核模式 `--verify-existing` 会在内存重新调用同一生产计算/保存函数，与既有npz逐数组一致后独立重算并重做8个错误注入；不会覆盖原边界文件，也不会训练或采集。cross-episode c 突变另在拼接序列上造成4/24超差，最大0.5550143867，证明检查能识别接到下一回合优势的问题。

生产 reset 合同的代码证据在本次冻结的 `runtime-code/src/task/wam-rl-policy-v3/tools/run/inspire_physics_host.py:190` 附近：post_physics_step 清零 reset_buf；`:236` 的显式 reset 才改变 episode；`:244` step 必须匹配所带episode。runner 的每步PD/clock合同还要求 episode不变、tick+1。不能由此推断其他自动reset环境也正确。

实际有bootstrap的time-limit回合：来源native episode=1、tick=600，保存observation前686维与原始轨迹最后一状态逐位一致；冻结的更新前critic重新前向，末状态价值与保存bootstrap差 **7.45058059692e-08**，在容差内。另一真正终止回合来源tick=207、episode=1，bootstrap=0，未查询末状态critic（对应observation检查为N/A，不能虚报通过一次查询）。两个回合的所有native episode在各自片段内恒定。

**覆盖自评：预算cutoff mask在本轮真实短跑中未被覆盖，joint safety也未在新PPO片段出现；它们在生产函数的确定性fixture中被覆盖，不是额外物理回合。** 本轮没有“靠随机初态凑齐终点”或延长到14,400 controls；G0b PASS由新真实落盘与明确构造的边界两层共同支持。

## v2 历史 R3：裁决及优势信号的适用范围

**v2既定范围裁决：G0b PASS，仅认证当时所验计算与软件边界落盘；最新物理覆盖裁决见文首及下节。** 原14,400 controls永久INVALID_INSTRUMENTATION；v1启动失败仍INVALID。均不因本次结果追认，也不把brief39的旧优势分布/信号读数冒充本轮新值。

“同recipe”仅限原计算表达式、γ/λ、Adam设置、lr=1e-5、两epochs、KL gate与相同checkpoint/数据来源/执行相关路径。由于只取两个初态并缩短采集，**新短跑不是原训练的等价结果**，不比较学习收益。单旧批次的逐位等价只证明该批计算未改变，不证明整段训练、GPU物理轨迹或其他buffer长度逐位重现。没有证据触发针对性较长复跑，本轮不扩预算。

新raw advantage分布来自807个实际保存样本：

| n | mean | std | P50 | P90 | IQR |
|---:|---:|---:|---:|---:|---:|
| 807 | -0.0619103837 | 0.3641086425 | -0.0863712654 | 0.4000180066 | 0.4805949554 |

截断bootstrap置零对照：`sum|A_boot−A_zero|=0.255695874626`，分母两版绝对量之和`476.658811415`，比值 **0.05364338%**；低于预注册50%触发线。分母定义与brief42一致（不是brief39的单版分母）。这里只报告反事实GAE差异，不推断critic已校准或PPO有效。

正优势共329/807。下表按action后实测阶段标志，阶段可嵌套、不是互斥分区；无阶段认证不等于无正优势：

| stage | 认证controls | 正优势controls | 该阶段内正优势比例 | 正优势和 |
|---|---:|---:|---:|---:|
| approach | 48 | 22 | 45.8333% | 7.0827064514 |
| contact | 2 | 2 | 100.0000% | 0.8864048719 |
| lift | 0 | 0 | N/A（零认证样本） | 0.0000000000 |
| hold | 0 | 0 | N/A（零认证样本） | 0.0000000000 |

## R4 落地：实际规模、源码身份、空间与复现命令

| arm | 初态/回合 | controls | 物理采样后更新 | worker elapsed |
|---|---:|---:|---:|---:|
| bc | 2/2 | 643 | 0 | 193.235s |
| ppo | 2/2 | 807 | 4 | 186.737s |

总物理controls **1,450**；PPO4 updates≤72，边界测试0controls/0训练。两个初态是原注册顺序前两项、native seeds294/295、model seed293；没有重新筛选初态或更换模型。旧注册的其他面板/成功判据文字仅作来源元数据保留，`g0b_scope`覆盖其研究目标，本轮不做其成功率比较或assessment。检查BC两个初态的native关节界、PD参数、offset/scale、DOF名称、dt与旧运行5项全部一致；不声称物理轨迹逐位重现。

两worker manifest记录git_commit不同（BC92563d9a、PPOfcc108e6），期间共享main发生他人提交；本轮实际runner.py与frozen GAE module在两臂**分别逐字一致**，SHA如下，实际代码以这两个文件及冻结runtime为准，不靠活动main的名字推定身份。对应证明见`v2/execution-receipt.json`。

```json
{
  "src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py": "97dd0562d3c266ea135a7e741160ff25e3c8ed262eaccb50beaee47034a1831f",
  "src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py": "2bf14bec975f66ab2f54cfbf06e060ef9fd13bced77ffbb508f968bbae713076",
  "patch_sha256": "9f621ada2502cfa1b4d213bfc17af602eaf596078f9345b48052a991ed5ca94d"
}
```

限定路径 diff（整体补丁含v1记录扩展及v2双mask；不含并行会话文件）：

```sh
git diff 64f04603^ 6cfb1d51 -- src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/check_boundaries.py --verify-existing
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/audit_new_rollouts.py
```

预期：边界`PASS cases=6 samples=24 mutants=8/8`；新数据`G0b PASS total_PPO_samples=807`，target/GAE/四种终止mask不一致均0。reader/边界脚本及audit_utils在输出目录，仅源码与本报告入库；scripts不产生新物理rollout。实际物理启动命令已在上节登记，supervise只成功执行一次v2，不可直接覆盖已存在目录。

所有4个native退出码0、两个policy client已退出，已知PID逐个存活检查均False；GPU7终态2MiB，其他卡未使用。无OOM/loader/非有限/磁盘故障；早前v1证据未删除。

### 提交并发的透明说明

v2源码提交6cfb1d51在共享暂存区碰到另一会话已暂存的5个文件，实际并入同一提交；另一会话随后推进92563d9a并推送。尝试只用独立index更正本地提交时HEAD已被推进，产生的临时amend已立即撤回，分支恢复到92563d9a并与远端核对，未强推、未覆盖工作文件；临时amend未进入远端历史。已发布历史不重写。后续本报告使用`git commit --only -- <本报告>`避免再次消费共享index；不能把6cfb1d51描述成仅含本轮文件的提交。原v1提交路径记录仅描述v1，不代表本次并发未发生。

v2产物快照 **22,681,641 bytes**，477个实体文件，约21.6309MiB；新attempt低于1GiB。运行结束磁盘余量约123.77GiB，旧v1的1.57GB单独保留，不把它当成本轮新增写盘。产物根为`outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v2/`，不入库。


## 08:20 新增要求：确定性物理边界覆盖预注册（v3，实施前）

最新指令要求通过收窄预算/选点得到真实物理终点，并把joint与普通终止分开各≥2例；此前v2的软件fixture不能代替该物理覆盖。本节新增物理数据与v2的807样本分开。原解读表阈值不变；v2计算PASS保留，扩展验收须等本节结果，覆盖不足则G0b INVALID（coverage），不以软件案例补足物理回合数。

只做以下6个预选案例，不随机凑数、不按结果追加候选：

| case | 原冻结初态 / native seed | 原臂与残差模式 | 收窄controls上限 | 预期物理终点 | 选点证据（引用） |
|---|---|---|---:|---|---|
| cut32 | s1_cylindersmall_pass_1 /294 | PPO原采样 | 32 | truncation，b=1/c=0，bootstrap非零 | v2同初态自然到600 |
| cut64 | 同上 /294 | PPO原采样 | 64 | 同上 | 同上 |
| joint-alarm | s10_alarmclock_pass_1 /295 | BC原均值 | 400 | joint safety，b=c=0 | v2 BC在43触发；旧BC在268触发 |
| joint-banana | s10_banana_pass_1 /296 | BC原均值 | 400 | joint safety，b=c=0 | 旧BC在346触发 |
| real-cube-1 | s10_cubesmall_inspect_1 /297 | BC原均值 | 400 | real termination（table），b=c=0 | 旧BC在224触发 |
| real-cube-2 | 同上 /297 | BC原均值 | 400 | 同上 | 同一冻结软件边界条件重复一次，不充作独立统计样本 |

只有cut案例主动缩短预算来保证早于已观测自然终点；普通/关节终止依据历史标签预选，GPU物理不能承诺逐位确定，必须用新实际终点验收。若选点产生别类终点，记“该构造未达到预期”，不改阈值/PD/奖励、不重试凑数。

所有案例从同一原始BC residual checkpoint启动，R/WAM、seed/data及原控制模式保持。BC不训练，先只补BC侧的GAE审计sidecar：使用已采集的obs/value/reward和生产GAE/保存函数，显式标“仅记录、未用于BC更新”。原PPO更新分支不变。新增仅为注册flag触发的落盘，不造新控制基线。

新增预算：BC≤1,600 controls、PPO≤96 controls；连同v2累计BC≤2,243、PPO≤903，均≤3,000/臂。PPO新增至多4 updates，总计≤8（上限72）；BC0更新。产物新目录`.../ppo-instrumented-audit-20261023-r1/v3-physical-boundaries/`，预计≤64MiB、硬上限512MiB。原v1/v2与旧14,400一律不覆盖、不删除。仍只GPU7，>2GiB等待60秒；OOM/loader/非有限/磁盘<20GiB立即停止不自动重跑。

无构造基线不是随机初态抽样：v2固定前两初态的PPO终点经验频率 real 1/2、joint 0/2、truncation 1/2；BC为real0/2、joint1/2、truncation1/2。真正随机初态总体出现率未取证，不能把这四回合当总体概率。旧brief42的36回合类别也仅可作固定面板历史频率引用，不作新G0证明。

状态：六个预注册v3物理案例已执行，无临时换点或追加回合；当前扩展验收INVALID（覆盖），结果见下。


## 确定性边界覆盖（08:20 追加要求的最终实测）

本节严格区分“想构造的类别”“实测终点”“按预期命中”。只通过选取原冻结面板初态和收窄预算，不改奖励、GAE、γ/λ、优化器、lr、收敛门、关节门或PD。每个case从同一原BC residual权重启动；2个PPO样本保持原随机采样与更新，4个BC样本保持原均值控制、0更新，只读记录GAE。

### 逐例台账

| sample id | 构造操作与理由 | 上限 / 实际controls | 预期 → 实际 | 末步b/c | bootstrap | 按预期命中 |
|---|---|---:|---|---|---:|---|
| cut32/cut32-A1 | 将原600时域的已观测长回合收窄到32 | 32 / 32 | truncation → rollout_cutoff | 1/0 | 0.116355091333 | 是 |
| cut64/cut64-A1 | 同一冻结初态收窄到64 | 64 / 64 | truncation → rollout_cutoff | 1/0 | 0.31010478735 | 是 |
| joint-alarm/joint-alarm-A1 | 选旧268/v2 43步joint初态，cap400 | 400 / 400 | joint_safety_termination → rollout_cutoff | 1/0 | 0.189175367355 | 否 |
| joint-banana/joint-banana-A1 | 选旧346步joint初态，cap400 | 400 / 99 | joint_safety_termination → real_termination | 0/0 | 0 | 否 |
| real-cube-1/real-cube-1-A1 | 选旧224步table初态，cap400 | 400 / 400 | real_termination → rollout_cutoff | 1/0 | 0.307653129101 | 否 |
| real-cube-2/real-cube-2-A1 | 同一冻结table条件重复一次，cap400 | 400 / 400 | real_termination → rollout_cutoff | 1/0 | 0.58116710186 | 否 |

32/64两例明确在已观测自然时域之前耗尽预算，实际均为cutoff且bootstrap非零，完成了主动收窄构造。其余历史终点标签未稳定复现：alarm到400仍未触发关节门；banana在99步先触发table gate；两次cube均到400被截断。因此关节预选0/2命中、真实终止预选0/2命中；banana的真实终止保留其实际标签，不伪装成按预期的关节构造。预期命中总计2/6。

### 三类终点的真实覆盖与逐样本mask

| v3新增实测类 | 终点样本/回合数 | controls | 应有末步b/c | mask不一致 | 解释 |
|---|---:|---:|---|---|---|
| 预算/时间截断 | 5 | 1296 | 1/0 | 0/1296 | 达到2例（其中主动cut构造2，其余为未命中其他目标后的实际截断） |
| 真实终止 | 1 | 99 | 0/0 | 0/99 | 仅1例，少1例 |
| 关节安全终止 | 0 | 0 | 0/0 | N/A（无样本） | 未覆盖，少2例 |

这里“样本”是一个终点transition/回合；不能把同一个终点之前的数百个continuing controls计成数百个边界案例。软件fixture的1个joint案例也不能补成物理joint案例。

按完整brief45已有仪器化新数据复用（v2 PPO + v3 BC/PPO），实际看到的类别如下；这只复用现有记录，不增加仿真、不重评分旧14,400：

| 全部有GAE sidecar的新数据 | 回合数 | controls | 距每类2例的缺口 |
|---|---:|---:|---:|
| 截断 | 6 | 1896 | 0 |
| 真实终止 | 2 | 306 | 0 |
| 关节安全终止 | 0 | 0 | 2 |

v2的BC曾有43步joint终止，但当时未保存BC的GAE/双mask，故它只能提供历史选点依据，不能追认成已完成逐样本仪器化joint验收。

### 不做构造时各类终点出现率：只能报告固定面板经验频率

| 未构造的v2原两初态（不是随机初态总体） | n | real | joint | truncation |
|---|---:|---:|---:|---:|
| BC | 2 | 0/2=0% | 1/2=50% | 1/2=50% |
| PPO | 2 | 1/2=50% | 0/2=0% | 1/2=50% |

**真正随机初态分布下的出现率：未取证。** 初态来自预定冻结顺序，不能把上述n=2当成随机总体率；没有为估计该率新增随机rollout，也不把v3人为缩短预算后的终点频率当自然发生率。旧brief42的36回合标签只作为已引用历史背景，不重算B2/B3、不追认其缺失字段。

### 计算链结果：新数据通过，覆盖门未通过

| v3新增逐样本检查 | 不一致/总数 | 最大绝对差 |
|---|---:|---:|
| target_return == value_t + advantage_raw | 0/1395 | 0.0 |
| 独立双mask GAE | 0/1395 | 5.15478786323e-08 |
| done/terminated/truncated/b/c与实测终点 | 各0/1395 | 布尔一致 |

预更新critic重建最大差2.68220901489e-07；末状态bootstrap重建最大差5.96046447754e-08。每个case原生episode恒定、末tick等于controls、截断observation对应reset前最后状态；所有样本reward分项之和与reward一致。对原始q/URDF界限及CAD/table几何独立复算终止原因，与记录一致；table原因不是凭文件名/预期类别推断。

BC新增raw GAE仅为审计值，不参与训练或BC损失；本节不会把它叫作“BC实际优化过的target”。各cohort分别报，不将短截断与普通回合混成学习效果：

| v3 cohort | 样本 | mean | std | P50 | P90 | IQR | 截断置零优势差占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bc | 1299 | -0.1189927713 | 0.3194521179 | -0.1467878968 | 0.3174773991 | 0.3171001896 | 1.65814215% |
| ppo | 96 | 0.1336320204 | 0.3052994063 | 0.1193088107 | 0.5392192900 | 0.3944479339 | 8.81275531% |

两cohort该比值均低于50%，未触发预注册critic校准线；这不是critic正确性或PPO有效性的证明。v3 PPO正优势62/96，BC审计正优势346/1299，但approach/contact/lift/hold认证均0，因此四阶段内正优势计数也均0；这是构造样本的观测，不外推性能。v2的807样本分布及阶段计数保持上文，不混算。

### 为什么未覆盖、最小额外规模与停止点

**当前单一裁决：G0b INVALID（物理关节类别覆盖不足），不是G0b FAIL（计算链）。** 阈值没有修改：已有数值比较都在rtol=1e-4/atol=1e-5内，但没有两个真实joint端点可以执行该类逐样本核对。原v2较窄范围的计算/软件边界PASS保留为历史事实，不顶替这次扩展验收；旧14,400仍永久INVALID_INSTRUMENTATION。

未覆盖原因是本次“历史终点标签→冻结初态选点”没有提供稳定的类别构造条件；具体物理分歧根因本轮未取证，不能径称PD、门阈或GAE的bug。现有生产reset只从该motion规定初态开始，没有在本轮允许范围内恢复旧关节终止前完整物理/控制历史的入口。没有更改门/PD去强行造终止，也没有随机换初态重试。

最小缺口明确区分口径：**若v3新增6例单独验收，还差1个真实终止+2个关节终止，至少3个有落盘的终点回合；复用v2已有真实终止后，完整brief仍至少缺2个真实关节终止回合。** 按预期构造的目标条件另未获资格（真实终止与joint预选均未命中），不能承诺重跑同点就能补齐。

实际需要的control数/保证覆盖的上界**未取证**；至少2个终点transition只是数学下界，不是可执行预算保证。若有可预先认证的joint reset点，下一次可预注册2例×≤400 controls（最多新增800，累计BC≤2742、PPO903，仍≤3000/臂）；但本轮尚无这样的已认证选点，**800不是保证覆盖的估计**。当前剩余预算不是继续随机凑数的理由。按追加要求在此收口，不启动长任务，不调lr/奖励/执行器。

### 补丁、身份、规模与复核

源码只改`run_bc_ppo_probe.py`：导入原`gae_targets`；`:300`新增默认关闭的`record_frozen_bc_gae`审计分支，`:312`调用原GAE，`:315`调用共享保存函数；物理结束后才读出原BC缓存并调用共享GAE/保存函数，0优化更新。原PPO更新if-body AST逐节点完全相同，GAE模块字节完全相同；因此此前297样本的6项逐位等价证明仍适用于未改的计算模块，不声称整个物理轨迹等价。所有6份预更新残差模型与原BC checkpoint逐参数相同；R/WAM仍为原checkpoint/step11000。

```json
{
  "src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py": "c0e57004d0815925e0207f731e5d3d490588480914211d69fe47c16a7cf0f773",
  "src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py": "97dd0562d3c266ea135a7e741160ff25e3c8ed262eaccb50beaee47034a1831f"
}
```

精确diff与复核命令：

```sh
git show 83f6d428 -- src/task/wam-rl-policy-v3/tools/run/run_bc_ppo_probe.py
# 已执行的唯一v3物理启动命令；现有目录不可覆盖，不要当作自动重试命令：
PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -u outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v3-physical-boundaries/supervise.py
# 只重算本轮已保存数据，不产生物理交互或训练：
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/v3-physical-boundaries/audit_physical_boundaries.py
```

预期读数：新增1395样本、calculation_errors为空、真实终点truncation5/real1/joint0、G0b INVALID。`plans.json`保留所有选点和上限，`supervisor.jsonl`逐例记录expected/actual与GPU等待；`preflight.json`记录原PPO AST与GAE字节不变、起始字节估计；每例`native-audit.json/trajectory.npz/critic-before-update.pt/ppo-rollout.npz`可独立核对。源码以六case冻结runner SHA一致为准，活动main期间他人提交不改变本轮冻结模块。

本次新增BC **1299 controls/0更新**，PPO **96 controls/4更新**；连同v2累计BC **1942**、PPO **903**，PPO总更新8≤72。BC审计不额外训练，没有新控制基线，没有重跑14,400。6个原生进程退出码均0、6个policy client均退出，审计存活检查全False；GPU7终态2MiB，其他卡未使用。无OOM/loader/非有限/磁盘故障，未删历史输出。

v3产物快照 **25,324,535 bytes**（527实体文件，约24.1514MiB），低于预估64MiB及硬上限512MiB；磁盘可用254.38GiB。outputs不入库。仍只显式路径commit；pull --rebase因他人脏文件被拒绝后未stash，普通push。
