看了最新的 [`64905eca`](https://github.com/hitsz-oyx/Ref2Dex-agent/commit/64905ecae1a32ac2999a06bbf01e95b288f30ea5)（10 月 17 日这次提交）。**这次有一个比“继续调 PPO、增大训练量”更优先的问题：当前 WAM adapter 的训练与评估执行了不同的策略——评估时把残差动作加了两遍。** 另外，WAM 历史窗口的重置也存在问题。

因此，我需要调整对最新结果的判断：**官方 PPO 的 warm-start 已经取得明显效果；但最新 WAM adapter 的负结果，需要先排除执行错误，才能归因于 WAM 或接入方式。**

这次我除了读实验卡，还按运行器的实际字符串替换逻辑，重建了它生成的上游网络代码。下面区分代码中能确认的问题、仓库报告的实验结果，以及仍需验证的解释。

## 一、最关键的代码问题：训练加一次 adapter，评估加两次

问题在 [`official_unidex_ppo.py` 第 286–293 行](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L286-L293)。

当前运行器通过字符串替换修改上游 `ActorCritic`：

1. 第 286–287 行的通用替换，在 `act`、`act_inference`、`evaluate` 三个函数里，各加入一次 WAM 残差。
2. 第 292–293 行，又在 `act_inference` 返回前加入一次相同残差。

我用仓库锁定的上游源码，执行了这段实际替换逻辑，结果是：

| 调用路径 | 用途 | 加入 WAM 残差的次数 |
|---|---|---:|
| `act` | 训练时采样动作 | 1 |
| `evaluate` | 计算 PPO loss | 1 |
| `act_inference` | 确定性执行、评估成功率 | **2** |

也就是说，训练时优化的动作均值是：

\[
\mu_{\text{train}}=a_{\text{native}}+r_{\text{WAM}}
\]

实际评估时却执行：

\[
a_{\text{eval}}=a_{\text{native}}+2r_{\text{WAM}}
\]

评估循环确实调用了 `act_inference`，所以这不是一段没有被使用的代码。[评估调用位置](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L161-L175)

### 这对目前结果意味着什么？

对无界 adapter，评估放大了它学到的修正动作。对 `gain=0.05` 的有界 adapter，每次残差虽然限制在 ±0.05，但重复相加后，**进入环境裁剪前的总残差上限变成了 ±0.10**。

所以，当前代码下的评估并没有测量训练时所优化的那个策略。对于接触、夹持、放下这类动作，额外施加一份修正，完全可能改变结果；但**修复之后能改善多少，现有数据还不能回答**。

### 为什么零初始化正控和 audit 都通过了？

因为 adapter 全零时：

\[
a_{\text{native}}+0+0=a_{\text{native}}
\]

因此，真正的 `model-initial.pt` 能恢复到 **65/128 success、39/128 held**，并不能检验非零 adapter 的训练与推理是否一致。仓库已经修正了此前误用 `model_0.pt` 做初始正控的问题，这一点应当保留；但正确的初始正控仍检查不出重复相加。[正控修正记录](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/docs/user/buglog/2026.10.17.md#L47-L69)

目前 `learning-audit` 主要检查参数变化、数值有限、优化步数等，也没有检查这三条路径的动作均值是否一致。[audit 代码](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L233-L243)

**我确认的是当前提交生成代码中的确定性错误。各次远端实验实际生成的 runtime 文件没有随这些实验卡提供，历史上哪些 run 使用了这份代码，还需要核对对应快照。最新 adapter 成功率应先按“评估实现待核验”处理。**

## 二、第二个问题：WAM 的历史窗口按整批环境清空，而且清空时点不对

当前 wrapper 在每次 `step` 后做的是：

```python
native, rewards, dones, infos = self.base.step(actions, id)
self._resets = bool(dones.any().item())
```

只要**任意一个环境**结束，`_augment` 就清空**所有环境共享的历史列表**，随后把当拍 snapshot 放进去。[wrapper 代码](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L88-L114)

这里有两个不同的问题。

### 1. 某个环境结束，会抹掉其他环境仍然有效的运动历史

WAM 使用的四帧历史来自最近七个仿真帧；历史不足时，代码重复最早一帧进行填充。刚清空后，四个历史位置都会指向当前这一帧。[历史采样代码](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/src/wam_rl_policy/fast_bc.py#L43-L50)

这意味着：即使环境 B 正在正常抓取，只要环境 A 结束，B 的运动历史也会突然变成重复帧。WAM 据此构造的速度、加速度等动态特征会发生人为变化。

训练又开启了初始 episode 时间错开。并行环境多时，整批历史被这种操作打断的风险更高；当前确定性评估则关闭了这种时间错开。**因此，训练和评估可能使用了很不一样的历史信息质量。** 实际清空频率和影响大小还需要日志确认，不能直接把它当作已经测量出的失败占比。[训练时间设置](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L29-L41)

### 2. 清空发生在终止帧，实际物理 reset 在下一拍

上游环境在当前 step 返回 `done`，但真正重置物体和手的状态，是在下一次 `pre_physics_step` 开头执行的。[上游 reset 时序](https://github.com/PKU-EPIC/UniDexGrasp2/blob/6c9e710b04769bf5d9765e3233820101cb3d718f/dexgrasp/tasks/shadow_hand_grasp.py#L910-L922)

因此，当前逻辑可能形成：

- 收到 `done`，清空历史，并写入旧 episode 的终止帧；
- 下一拍实际 reset；
- 若没有再次触发整批清空，新 episode 首帧被追加到旧终止帧后面。

这样，WAM 会把**物理重置造成的瞬间跳变**当作连续运动的一部分。

这项问题同时涉及 **WAM 直接拼接和 adapter** 两条路径。它使“WAM 特征在 PPO 里没有帮助”这个解释多了一个必须先排除的因素：输入的时间连续性本身可能没有保持好。

## 三、官方 PPO 的新结果，确实让判断前进了一步

上一轮“官方 PPO 还没有完整结果”的判断，现在需要更新。新实验卡已经给出了以下结果：

| 策略起点 | 新增训练量 | TRAIN：success；held | DEV：success；held |
|---|---:|---:|---:|
| 随机初始化，官方 PPO | 81,920,000 环境步 | 0/128；0/128 | 0/128；0/128 |
| 官方发布 checkpoint 直接执行 | 无新增训练 | 65/128；39/128 | 此处未报告 |
| 官方 checkpoint 初始化，再继续 PPO | 81,920,000 环境步 | **103/128；85/128** | **98/128；72/128** |

这里 `held` 指实验卡中的 `held60_with_lift10cm`。来源：[fresh 与发布模型结果](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/docs/experiments/probes/P-20261016-official-unidex-ppo.md#L84-L109)、[warm-start 结果](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/docs/experiments/probes/P-20261016-official-unidex-ppo.md#L112-L120)。

这组结果支持两个重要判断。

**第一，当前环境里，已有抓取技能能够通过 PPO 继续提高。** 所以，把问题笼统归为“PPO 在这里完全学不动”，已经不准确。

**第二，从零形成抓取技能仍然没有复现。** 同样的交互量下，初始化差异对应着很大的任务表现差异。这更支持优先检查从零探索、奖励激励和完整训练配方，而不是只盯着优化器是否更新。fresh run 的 critic EV 约为 0.905，但成功率仍为零，也说明“能拟合训练中收到的回报”不等于“学到了成功抓持”。

至于到底是探索不足、奖励存在容易停留的行为，还是与官方完整训练条件仍有差异，**这组结果还不能进一步区分**。

另外，warm-start 成功不等于仓库 Objective A 已完成：它仍以官方 actor checkpoint 为策略基础。它可以作为有能力的诊断底座，但需要明确它回答的是哪一个研究问题。

## 四、最新 WAM 负结果应收紧解释，不能直接定性为表征失败

实验卡新增的几个主要结果是：

| 设置 | 评估面板 | WAM：success；held | native 对照：success；held |
|---|---|---:|---:|
| 无界 adapter，64 env × 1000 iter | TRAIN | 1/128；0/128 | — |
| 无界 adapter，256 env × 1000 iter | TRAIN | 2/128；0/128 | — |
| 有界 adapter，256 env × 1000 iter | TRAIN | 65/128；37/128 | 73/128；61/128 |
| 有界 adapter，1024 env × 300 iter | DEV | 45/128；23/128 | 69/128；45/128 |

来源：[adapter 实验记录](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/docs/experiments/probes/P-20261016-official-unidex-ppo.md#L145-L182)。

**这些是仓库报告的结果；当前代码审查发现的问题，使它们暂时不足以证明训练得到的 adapter 策略本身就有这么差。**

除此之外，这里的“同预算 native 对照”还有一个归因限制：

- WAM adapter 分支冻结了原 actor、critic、`log_std`，只训练新增线性 adapter。
- native-only 分支更新原网络。

所以，比较中同时改变了 WAM 信息和可训练模块。即使修复评估，仍不能把全部差距都归给 WAM 特征。冻结 critic 是否限制了策略更新，也是优化设计上的疑点，**它与前面确定存在的重复加法 bug，应当分开讨论**。[冻结代码](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/src/task/wam-rl-policy/tools/run/official_unidex_ppo.py#L273-L282)

### 文档还有一处需要清理的矛盾

最新 STATE 和实验卡仍写着：

> official initialized `model_4000.pt` 为 103 success、84 held、108 lifted。

但同一个提交的 buglog 明确指出，当时没有这个 official step4000 的对应评估产物，不能把最终结果当作同 iteration 对照；也没有看到像初始正控那样的后续补测说明。[buglog HIGH-004](https://github.com/hitsz-oyx/Ref2Dex-agent/blob/64905ecae1a32ac2999a06bbf01e95b288f30ea5/docs/user/buglog/2026.10.17.md#L93-L108)

这组“官方 4000 对照”应先剔除或标为待核验。WAM `model_4000` 的 10/128、6 held 可以作为已报告的截断 checkpoint 结果，但不能给它配一个证据冲突的官方对照。

## 五、现在最值得做的下一步很具体，而且第一步不需要重训

**我建议先把现有 checkpoint 的真实能力测准，暂缓追加同配方长训练。**

1. **修正 adapter 的训练／推理一致性，再重评已有 checkpoint。**  
   统一三条调用路径的动作均值计算，用一个**非零 adapter** 和相同 observation，检查 `act` 的分布均值、`evaluate` 中的均值与 `act_inference` 一致。先只修这个错误重评，就能直接回答：“之前掉点有多少来自评估多加了一份残差？”这一步不需要重新训练。

2. **将 WAM 历史改为逐环境维护，并对齐实际 reset 边界。**  
   至少验证两件事：一个环境 reset 不影响其他环境历史；任何一个 H4 窗口都不跨 episode。这里修正的是训练输入，如果旧训练已经受到影响，单纯重评旧 checkpoint 不能完全补救，之后才考虑必要的小预算重训。

3. **在上述问题解决后，再判断 WAM 的边际作用。**  
   对照应保持相同的可训练模块和优化条件，只改变是否提供有效 WAM 信息。否则，结果仍混合了“表征差异”和“策略能以什么方式更新”的差异。

这些新问题主要针对这次官方 PPO 接 WAM 的实现。上一轮已经看到的短期目标语义错配、控制泛化不足、coordination 奖励没有稳定保持收益，仍是其他实验链上的独立问题，不会因为发现这两个 bug 就自动消失。

**我现在对“为什么效果不好”的优先判断是：最新这一批实验，已经出现了可以直接定位的执行错误和历史输入问题，应先解决它们；修复后若 WAM 仍然没有收益，再回到表征与控制目标是否匹配的问题。现在继续加环境、加时长，会把这几种原因继续混在一起。**

## 2026-10-17 修订：先做实现校正 Probe

凌晨后的证据让上述判断需要两处收紧。第一，官方 initialized PPO 的 `103/128` TRAIN、
`98/128` DEV 是 warm-start 诊断，说明当前环境和奖励允许已有技能继续提升，但不完成
Objective A；fresh 从零训练仍是 `0/128`，两者不能混称官方 baseline。第二，WAM 的
`model_4000` 仍可作为 WAM 截断 checkpoint 的描述性结果，但缺少可信的同轮 official
评估产物，删除其 `103/128, 84 held` 对照，直到能从独立 JSON 复核。

代码修订先落实两个最小合同：adapter residual 在 `act`、`evaluate` 和
`act_inference` 各计算一次；WAM H4 history 按环境保存，终止帧只用于当前返回值，下一拍
只清理已终止的环境。旧训练结果保留为“旧 runtime 待核验”，不作为修复后策略的证据。
修订后的远端 Probe 先做生成 runtime 的动作路径审计和旧 checkpoint 重评，再做
`gain=0.05`、64-env、1000-iteration 的短训练；若审计失败、出现非有限值，或连续八个
完整物体段的 critic EV 近零，立即停止。只有在修复后结果与训练动作语义一致时，才继续
比较 WAM 与 native 的边际收益。
