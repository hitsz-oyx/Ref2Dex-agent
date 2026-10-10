# ref1_1：PointWAM 原始来源核对

日期：2026-10-10。只读核对 [ref1_1.md](../user/ref/ref1_1.md)，不含实现、训练或仿真。

已先读 [既有 schema](../../../../../paper/pointwam/schema.json) 和 consequence-evaluator 的 [候选 τ 文献笔记](../../../consequence-evaluator/docs/research/20261010-candidate-tau-proposal-review.md)。本次刷新作者页面与论文 v1，不新增第二份论文索引。

## 可直接追溯的事实

下表保留紧凑英文记录以限制对同一论文的摘述量；链接指向支持对应事实的原文节。

| 范围 | Paper-reported facts | 来源 |
|---|---|---|
| 输入与 world modeling | Colored scene, current hand points, instruction jointly predict hand/scene displacements; candidate actions are not inputs. | [§3.1–3.2](https://arxiv.org/html/2610.02840v1#S3.SS1) |
| Retargeter | A two-block Transformer decoder reads predicted hand displacements and current robot state, outputting parallel chunks. Scene forecasts do not enter it. | [§3.3](https://arxiv.org/html/2610.02840v1#S3.SS3)、[B.4](https://arxiv.org/html/2610.02840v1#A2.SS4) |
| 监督 | Robot fine-tuning jointly trains forecaster/retargeter with trajectory squared error and recorded-action L1. No detach ablation is reported. | [§3.4](https://arxiv.org/html/2610.02840v1#S3.SS4) |
| 动作合同 | DexJoCo: 22 coordinates/arm, including 16 finger targets; state-relative targets are standardized and inverted at inference. | [B.5](https://arxiv.org/html/2610.02840v1#A2.SS5) |
| 数据规模 | 1,148,324 human episodes; 959 replay-successful robot demonstrations; 60k fine-tuning iterations. | [§4.1.1](https://arxiv.org/html/2610.02840v1#S4.SS1.SSS1)、[C.1](https://arxiv.org/html/2610.02840v1#A3.SS1) |
| Scene ablation | Volt-L12/EgoDex: 64.4% versus 53.5%; scene supervision removed during both pretraining and fine-tuning; three evaluation seeds. | [§4.4/Table 3](https://arxiv.org/html/2610.02840v1#S4.SS4) |
| 执行时序 | 30 predictions, 20ms/control-step; asynchronous requests when fewer than 24 remain; empty buffers hold pose. | [C.2](https://arxiv.org/html/2610.02840v1#A3.SS2) |
| 限制 | Retargeter imposes no explicit joint/collision constraints; authors characterize recorded targets as respecting both. | [B.4](https://arxiv.org/html/2610.02840v1#A2.SS4) |

源码边界：直接读取 [官方项目页](https://chrockey.github.io/PointWAM/) 仍见 `Code soon`，Code 按钮的 `href="#citation"`、`aria-disabled="true"`；作者的 [publication metadata](https://chrockey.github.io/data/publications.json) 对该论文只列 arXiv/Website，没有源码链接。GitHub API 查询遭遇 rate limit，不能据此断言不存在任何仓库。当前只能说**尚未找到作者发布、可核实的实现**；不把同名非作者仓库当作官方代码。

## ref1_1 的支持、未证与需要修正之处

- **支持到论文设计层：**“τ 是 intermediate action representation”“由 robot recorded actions 监督 learned retargeter”“scene prediction 监督共享 representation”都有直接依据。
- **梯度链尚未做源码实证：**按论文方程，在没有 detach 时 action loss 可以经过 predicted τ 更新 forecaster；联合目标与可微模块支持这种实现解释。但没有作者代码可检查实际 detach/参数组，也没有独立梯度干预对照。应写“按公开设计可回传”，不能写“已审计实际回传，且证明这正是成功原因”。
- **PointWAM ≠ action-conditioned PointWorld：**本项目若要 `(H,候选 τ)→E/z_phys`，仍须独立建立这份条件数据合同。联合 co-evolution 预测不能自动提供“同一状态换任意候选动作”的预测接口。
- **Recorded ≠ 我们的安全保证：**作者对其目标的描述不证明我们的 native action 标签合法，更不保证 learned 输出合法。要明确日志存的是 latent、clamped command 还是最终 PD target，并独立审计单位、限幅、耦合与执行时对齐。抓取本身还需要接触，不能笼统把“无碰撞”当作要求。
- **早发请求 ≠ 固定六步闭环替换：**少于 24 这一触发条件表示至少已消耗七项后才会发请求；新结果生效还受异步延迟和 buffer 合并规则影响。缺少源码时不能直接声称“每六步新观测立即替换旧 chunk”。因此 ref1_1 的“8 步不是主要问题”是本项目待检验判断，论文不能排除它。
- **Data scale/ablation 不定位我们的故障：**论文条件与本项目不同。10.9pp 支持特定 recipe 的 scene-supervision utility，不能替代本项目 native action / τ baseline、executability 或 Cm-on/off 的证据。

## 对新 Task 的最小含义

值得试的是新的监督合同，而不是声称已经复现整套 PointWAM：`(s_t, actual τ_(t+1:t+24)) → actually executed A_(t:t+K−1)`。普通完整 rollout 可以提供这类配对；只做监督拟合不必先建立 fork。

但“逆着组织数据”不保证映射唯一：近似同一手轨迹可能由不同 preload、阻抗或外部约束产生，限幅还可能使不同 command 得到近似相同运动。第一轮应检查轨迹相似而动作标签冲突的样本，并区分 GT-τ 输入下的能力与 predicted-τ 输入下的能力；后者失败不能仅由前者的回归 loss 解释。

输入 τ 必须来自同一实际轨迹，未来动作标签必须是同一窗口的实际执行值；按 episode 分 split，不能把相邻滑窗随机拆分当泛化。成功与失败都可作为 inverse supervision 数据，但失败数据不会自动教会任务意图。

数据存在还不等于训练可用：本会话 root 已核对旧 teacher packet 标有 `engineering_only=true`、`training_allowed=false`，必须排除，不能因其成功而转作示范训练。新 Task 的 rollout 需要按实际 reset/terminal 边界切 episode，并排除缺失最终状态或跨 reset 的未来窗口；几百相邻滑窗仍只来自一条 episode。是否包含成功轨迹与独立 episode 数，应由单独数据合同审计决定。

普通 rollout 足够构造监督样本，不代表足够验证 τ 的控制作用。最小诊断应含 state-only、τ shuffle；在相同初始条件更换真实可执行目标并观察 hand tracking，才能区别“依靠 s 复制旧策略”与“按照 τ 控制”。这一步是后续有预算的 Probe，本次不运行。

如果实现 learned retargeter，应单独记录 GT/predicted τ 训练比例、是否 action loss 到达 forecaster、以及部署时是否每步重新读取状态。输出 native action chunk 的网络不天然继承旧 R 的逐步状态反馈。以上是本项目需要检验的合同与假设，不是论文已替我们证明的结果。
