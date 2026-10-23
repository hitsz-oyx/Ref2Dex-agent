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

状态：预注册已先落盘，随后完成仅持久化补丁与 CPU 等价性核对；物理复跑待启动。

## 补丁与运行前等价性证据

`src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/intent_residual.py::ppo_update` 增加可选 `audit` 接收器；在原始 GAE 算完、标准化之前复制 `advantage_raw`、实际进入 value loss 的 `target_return`、`V_old`、reward、next value、γ/λ 和末状态 bootstrap 值。原计算表达式、PPO 损失、Adam、KL gate、两 epochs 均未修改。runner 仅把这些副本与每样本 done/bootstrap/终点标签及已存在的实测阶段一起写入 `ppo-rollout.npz`。

`done` 是回合边界标志（所有回合最后一个样本为 True）；`bootstrap` 是一步 TD 的 next-value 有效标志（中间样本为 True，真正终止/关节安全终止末样本为 False，时间/预算截断末样本为 True）。`termination_type` 中间样本为 continuing，末样本为三类之一；GAE trace 在任何 done 处停止递推，但截断 TD 仍使用末状态 value。这些字段描述原 runner 的整回合计算，不反过来控制更新。

等价性：使用旧 PPO `p0000-A1/ppo-rollout.npz` 的 297 样本、同 BC checkpoint、seed 293，在 CPU 上分别调用原冻结函数与补丁函数各一次（各 2 optimizer updates）。6/6 比较逐位相同：raw advantage、target、标准化优势、全部更新指标、模型 state_dict、Adam state_dict；最大 raw advantage 差=0、最大模型参数差=0。两 epoch loss 两边都是 `[0.08426089584827423, 0.06399012356996536]`。这是同 CPU 批次的严格等价，不声称 CPU 与旧 GPU 浮点执行逐位相同。原 GPU 数据在 CPU 的行为 logprob 重建最大差 2.288818359375e-5，低于原 gate 1e-4。

```sh
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' /home2/wyy/miniconda3/envs/graspenv/bin/python outputs/wam-rl-policy-v3/ppo-instrumented-audit-20261023-r1/check_equivalence.py
```

预期：`equivalence.json::status=PASS`，所有 checks=true，两侧 updates=2，raw/参数最大差=0。该核对不新增环境交互。

运行环境：policy Python3.8.20 / torch2.4.1+cu121 / NumPy1.24.4，native 使用原 Python3.8.10 / torch2.0.1+cu118 环境。冻结旧 PPO runtime，唯一算法包差异为上述落盘副本；原 R、v6、BC residual 权重与数据源均保持。输出根 scratch 替代 runner 原来的仓库 tmp，以符合本轮写入边界；路径超过 AF_UNIX 的 108-byte 限制时，输出目录内 `short_unix.py` 只在 socket bind/connect 系统调用处使用同一文件的 `/proc/self/fd` 短别名，137-byte 路径的原样 payload 往返已通过。native wrapper 仅安装此传输适配，不修改物理或模型代码。实际子进程命令另存 `*-native-commands.jsonl`。

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
