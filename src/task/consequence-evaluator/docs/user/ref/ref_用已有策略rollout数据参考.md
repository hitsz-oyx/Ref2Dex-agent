对，你记得的名字大概率就是 **DART**，不是 DRFT。我把 DART 的源码、近年的 recovery 数据采集、以及你们当前 `collect_continuous.py` 对着看了一遍。结论是：

**“整条 rollout 从头到尾加噪”现在并不是最适合你们的方案。你们真正该借的是 `DART 的扰动分布学习 + CARE/RaC 的阶段触发采集`。**

### 先说 DART：你记忆没错，但它不完全解决你的问题

DART 全称 **Disturbances for Augmenting Robot Trajectories**。核心思想就是：

\[
a_t^{exec}=a_t^{expert}+\epsilon_t
\]

而且不是拍脑袋定噪声大小，它根据当前 learner 和 expert 的动作误差估计协方差：

\[
\Sigma \approx
E[(a^{expert}-a^{learner})(a^{expert}-a^{learner})^\top]
\]

再用这个 \(\Sigma\) 给 expert rollout 加 Gaussian noise，使 expert 主动访问 learner 将来可能跑偏到的区域，从而产生 recovery 数据。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v78/laskey17a.html?utm_source=chatgpt.com)

源码现在还在：

[Berkeley DART 源码](https://github.com/BerkeleyAutomation/DART?utm_source=chatgpt.com)

我看了它的 `supervisor.py`，逻辑非常直接：

```python
intended_action = policy.intended_action(s)
sampled_action = np.random.multivariate_normal(intended_action, cov)
```

也就是每次调用 expert action 都采一次噪声。`noise.py` 再通过 learner/expert action difference 更新 covariance。

所以你的担忧是成立的：

> DART主要回答“扰动多大、扰动各维怎么相关”，但没有很好回答“在哪个阶段扰动才有信息价值”。

比如你们 airplane：

```text
approach → contact → grasp → lift → hold
```

在 approach 很早期一直加手指扰动，很可能只是产生一堆“离物体还很远”的无意义数据。

---

## 真正和你想法最接近的是 CARE

这个是我这次搜到**最值得你们直接参考的**。

CARE 是 2026 年 9 月公开的工作，它明确不是：

> 随机到处加 perturbation。

而是：

\[
\boxed{
\text{先运行 nominal policy}
\rightarrow
\text{收集真实 failure}
\rightarrow
\text{按 stage 统计 failure distribution}
\rightarrow
\text{在对应 stage 人工复现这种 failure}
\rightarrow
\text{收集 correction}
}
\]

论文明确强调它使用 **stage-conditioned post-failure deviations**，而不是 manually designed/random perturbations。[Hugging Face](https://huggingface.co/papers/2609.24118?utm_source=chatgpt.com)

[CARE 源码](https://github.com/xiaojunlan/care?utm_source=chatgpt.com)

而且它真的有工程代码，不只是论文。

我专门看了：

`envs/put_object_cabinet_regrasp.py`

它做的事情非常贴近我们的场景。

正常运行到**抓取阶段**以后，它才制造一个错误抓取：

\[
(\Delta x,\Delta y,\Delta z,\Delta yaw)
\]

而这些 offset 不是简单 uniform random，源码里已经写了不同维度的拟合分布，例如 Gaussian、generalized normal、skew normal、Weibull 等，然后从大量候选中按密度取一个 representative bank。

然后执行：

```text
正常到达抓取阶段
      ↓
加入错误 grasp offset
      ↓
执行错误抓取
      ↓
进入 failure state
      ↓
退开
      ↓
重新计算正确 grasp
      ↓
执行 recovery
```

更有意思的是，源码里：

```python
self.move(wrong)
self.start_record()
```

也就是说，**前面正常走的那一大段甚至可以不作为 correction 数据记录**。

这正是你刚才说的：

> “如果从一开始就加扰动，可能会产生很多没必要的地方。”

CARE 就是在解决这个。

但我也要提醒一个工程现实：CARE 当前仓库只有一个公开提交，是一个 code snapshot；README 自己承认**failure distribution fitting 命令还没有完整提供**。当前代码已经把拟合好的参数写进采样函数，但“从 nominal failed rollout 自动拟合这些分布”的完整 pipeline 还需要补。[GitHub](https://github.com/xiaojunlan/care?utm_source=chatgpt.com)

所以它是：

**思路极适合我们，代码非常值得抄，但不是开箱即用。**

---

## 另外一个非常成熟的方向：RaC / DAgger

这一类干脆不主动到处加噪声，而是：

\[
\text{policy 正常运行}
\]

发现：

\[
\text{“马上要失败了”}
\]

才进入 recovery 数据采集。

RaC 的协议很明确：

\[
\text{Autonomous rollout}
\rightarrow
\text{failure imminent}
\rightarrow
\underbrace{\text{Recovery}}_{\text{回到合理状态}}
\rightarrow
\underbrace{\text{Correction}}_{\text{继续完成当前子任务}}
\]

而且 intervention 完成以后直接结束 episode，避免后面的数据受到 human/policy 混合分布污染。[RaC Scaling Robot](https://rac-scaling-robot.github.io/?utm_source=chatgpt.com)

现在最重要的是：

### HuggingFace LeRobot 已经把这套做成正式工程代码了

不是论文伪代码。

[LeRobot DAgger 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/dagger.py?utm_source=chatgpt.com)

它已经是完整状态机：

```text
AUTONOMOUS
    ↓
PAUSED
    ↓
CORRECTING
    ↓
PAUSED / finish
```

而且支持：

```text
record_autonomous=True
```

记录整条；

或者：

```text
record_autonomous=False
```

**只记录 correction window。**

也就是说：

> policy 正常跑 300 步都没事 → 一帧都不用存 correction dataset；
> 第 301 步快掉了 → takeover；
> 只记录接下来 20～50 步恢复动作。

这套工程基础其实非常成熟了。[GitHub](https://github.com/huggingface/lerobot/blob/main/src/lerobot/rollout/strategies/dagger.py?utm_source=chatgpt.com)

---

## 还有一个更激进、特别适合仿真的：SimDAgger

这个项目虽然没有 DART/LeRobot 那么“经典”，但工程结构跟我们非常像。

[SimDAgger 源码](https://github.com/ClubbedBHS/SimDAgger?utm_source=chatgpt.com)

它不是固定：

```text
第 200 步 perturb
```

而是在线检测：

```text
当前 task phase
+
task progress
+
contact/geometry
+
进度是否停滞
+
是否发生 drop/rebound
```

然后才决定：

\[
\text{takeover}
\]

源码甚至有这样的流程：

```text
policy 正常 rollout
        ↓
online failure onset detector
        ↓
检测到 stall / drop / knock
        ↓
记录 trigger state
        ↓
expert recovery
```

它还有三个模式：

| 模式 | 行为 |
|---|---|
| immediate recovery | 一检测到 failure onset 就 recovery |
| detect only | 不干预，只统计 detector 对不对 |
| confirmed recovery | 先完成 rollout 确认真的失败，再回到 trigger 附近收 recovery |

并且代码把逻辑拆成：

```text
task_progress
      ↓
failure detector
      ↓
takeover trigger
      ↓
recovery controller
```

这个模块化设计非常适合抄。[GitHub](https://github.com/ClubbedBHS/SimDAgger?utm_source=chatgpt.com)

不过其中“snapshot → rewind → branch”的部分**不适合直接搬到你们 Isaac Gym 正式数据采集**。你们之前已经碰到完整状态恢复的问题；所以我只会借它的 **phase tracker + failure trigger**，不会借它的 rewind 数据生成方式。

---

## 还有 REIM：可以参考“如何制造 failure”

REIM 的代码也公开了：

```bash
python scripts/generate_failures.py
```

它在 policy rollout 时注入三类扰动：

\[
\text{action disturbance}
\]

\[
\text{object disturbance}
\]

\[
\text{observation disturbance}
\]

并且记录“未来一段 horizon 内是否发生 failure”，用于训练 causal failure detector。[GitHub](https://github.com/CC-robotics/REIM?utm_source=chatgpt.com)

这对我们有一个重要启发：

**不要只扰动 action。**

我们以后如果真的想让 evaluator 泛化，可以区分：

```text
action error
execution/control error
object disturbance
contact disturbance
```

但第一版没必要这么复杂。

---

# 回到我们的仓库：其实已经做对了一半

我刚又看了你们现在的：

`src/task/consequence-evaluator/tools/run/collect_continuous.py`

它已经不是 DART 那种“从头到尾加噪”。

现在已经有：

```text
assigned_phase
perturbation_tick
target_phase
residual_plan
```

而且采集合同本身就是：

```text
正常 expert rollout
       ↓
到 assigned phase
       ↓
启动一次 24-step residual plan
       ↓
residual 结束
       ↓
恢复 expert
       ↓
继续到 episode end
```

所以这一部分**方向其实是对的**。

你们现在更像：

\[
\boxed{\text{简化版 CARE}}
\]

而不是 DART。

当前真正的问题在于：

### 我们的“什么时候扰动”已经不错

例如：

```text
grasp
lift
hold
```

才启动 perturbation。

### 但是“扰动什么、扰动多大”目前还比较粗糙

现在主要还是：

```text
--amplitude 0.08
```

然后生成 smooth residual。

这个分布其实没有回答：

> baseline 真正失败的时候，它通常偏到哪里？

这恰好是 CARE 和 DART 可以补给我们的部分。

---

# 我认为最适合我们的完整方案

我现在会把这几套东西组合起来，而不是选某一篇全抄。

第一轮**完全不加人工扰动**：

\[
\pi_0
\rightarrow
\text{大量自然 rollout}
\]

只统计真实 failure。

在每个阶段记录：

\[
\text{phase}
\]

\[
H_t
\]

\[
a_t
\]

\[
E/I
\]

以及失败前的偏差，例如：

\[
\Delta p_{wrist},
\Delta R_{wrist},
\Delta q_{finger},
d_{hand-object},
v_{object},
...
\]

得到：

\[
p(\Delta \mid phase,\ failure)
\]

这就是 **CARE 思路**。

例如最终发现 airplane 的真实失败主要是：

```text
grasp:
  thumb yaw 偏小
  wrist x +12 mm

lift:
  wrist rotation 偏差
  object vertical velocity 开始下降

hold:
  finger curl 不足
```

然后第二轮才做 targeted rollout：

\[
\text{正常执行}
\]

到：

\[
phase=\text{grasp}
\]

以后再从：

\[
p(\Delta|\text{grasp failure})
\]

采一个 perturbation。

不是：

\[
\Delta\sim U(-0.08,0.08)
\]

而是：

\[
\boxed{
\Delta\sim
p(\Delta|\text{真实失败分布})
}
\]

这就是 CARE。

---

然后再借一点 DART：

CARE 学“失败长什么样”。

DART 学“当前 policy 和 expert 差多少”。

所以扰动 covariance 可以进一步根据：

\[
e_t=a^{expert}_t-a^{policy}_t
\]

调整。

最后变成：

\[
\Sigma_{\text{phase}}
=
\operatorname{Cov}
(a^{expert}-a^{policy}\mid phase)
\]

于是：

\[
\boxed{
\delta
\sim
\mathcal N(
\mu_{\text{failure,phase}},
\Sigma_{\text{phase}}
)
}
\]

这会比现在固定 amplitude=0.08 合理很多。

---

## 我会把采集器改成这个状态机

```text
            reset
              │
              ▼
        baseline/expert
        正常连续执行
              │
              ▼
       phase detector
              │
       ┌──────┴──────┐
       │             │
 非关键阶段       关键阶段
       │             │
继续正常执行          ▼
               trigger gate
                    │
          ┌─────────┴─────────┐
          │                   │
       clean               perturb
          │                   │
          │            sample δ from
          │          phase failure dist.
          │                   │
          │            execute K steps
          │                   │
          └─────────┬─────────┘
                    ▼
             baseline recovery
                    │
                    ▼
              episode end
                    │
                    ▼
                 label
```

注意：

**一个 episode 最多扰动一次。**

这一点我反而建议保留你们当前设计。

因为如果：

```text
grasp 扰动一次
lift 又扰动一次
hold 再扰动一次
```

最终 outcome 就很难 attribution：

> 到底是哪次 perturbation 导致成功/失败？

而我们现在要训练 evaluator：

\[
(H,\delta,Z)\rightarrow Y
\]

需要因果关系尽量清楚。

所以：

\[
\boxed{\text{一条 rollout，一个关键 perturbation}}
\]

非常合理。

---

# clean 数据也不能删

我不会像纯 correction dataset 那样只保存失败附近。

因为我们的目的不是训练 recovery policy，而是训练 evaluator。

Evaluator 必须看到：

\[
\delta=0
\]

或者正常 action 的 consequence。

所以建议每个阶段都保留一部分：

```text
clean      30%
mild       30%
medium     25%
strong     15%
```

具体比例先别定死，但需要**有 clean control group**。

这样 evaluator 才知道：

\[
\text{“不干预”}
\]

和

\[
\text{“小干预”}
\]

哪个更好。

---

# 所以几套方法我现在会这样评价

| 方法 | 什么时候扰动 | 扰动怎么定 | 对我们价值 |
|---|---|---|---|
| **DART** | 几乎全程 | learner/expert error covariance | 学“噪声多大”，但不解决阶段 |
| **HG-DAgger / RaC** | 即将失败才介入 | 人工 recovery | 学“什么时候值得采” |
| **LeRobot DAgger** | 人触发 | correction window | 工程框架最成熟 |
| **CARE** | **任务关键阶段** | **真实 failure distribution** | **和我们最匹配** |
| **SimDAgger** | 自动 failure onset | task progress / geometry | trigger 模块很好抄 |
| **REIM** | rollout 中注入 | action/object/obs disturbances | 可做后续扩展 |

所以如果你问我：

> **有没有已经成熟到能指导我们现在怎么做的？**

有。

我现在不会再设计“从 episode 0 开始随机加扰动”了。

我会直接定成：

\[
\boxed{
\textbf{CARE-style stage-conditioned perturbation}
+
\textbf{DART-style adaptive magnitude}
+
\textbf{RaC/SimDAgger-style trigger}
}
\]

而且你们现有 `collect_continuous.py` **不用推倒重来**。最大的修改只有两个：

1. 把现在固定的 `amplitude=0.08` 换成 **phase-conditioned perturbation bank/distribution**；
2. 把现在较硬的 phase trigger 逐步升级成 **phase + failure-risk trigger**。

也就是说，之前那些 `approach/contact/grasp/lift/hold` 和“只施加一次 24-step residual”的工程都可以保留。

我认为下一步最值得做的不是马上再采一批，而是**先用现有正常/失败 rollout 统计 airplane 在 grasp/lift/hold 三个阶段真实失败前的 residual/state 偏差分布**。如果这个分布能提出来，你们的下一版 collector 就可以非常接近 CARE 的成熟路线，而不是继续拍一个统一噪声方差。