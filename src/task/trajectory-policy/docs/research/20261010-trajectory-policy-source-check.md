# trajectory-policy 原始来源核对

日期：2026-10-10。范围：核对 [ref1.md](../user/ref/ref1.md) 中的 RLT 类比、RL 梯度与 chunk 时序；只读研究，没有训练或仿真。以下区分论文事实与本项目推论。

## RLT 可以借鉴什么

[RL Token 原论文 v1，§IV-A/B、式 1–5](https://arxiv.org/html/2604.23073v1#S4) 的 token 压缩冻结 VLA 的观测表征；actor 另接本体状态和已采样 reference chunk，输出完整 Gaussian action chunk。它通过动作 L2 正则靠近 reference，并采用 reference dropout；并没有强制写成 reference 加 residual。token 也不是候选动作产生的物理后果。因此，把本项目的 `F(H,τ)` 称为 RLT 原结构，或把 RLT 描述为永久 residual，都不准确。

本项目推论：借鉴“冻结大模型提供紧凑表征、小 actor-critic 做 RL”是合理设计输入；`z_phys = F(H,τ)` 则是不同的、尚待验证的动作条件特征。其抓取收益不能由 RLT 的实机结果直接推出。

## PPO 与动作条件 Q 要分别落地

[PPO 原论文，式 7、9](https://arxiv.org/pdf/1707.06347) 用采样动作的概率比和 advantage 更新 actor，并学习状态价值。以 `c` 为随机动作时，应保存采样的 `c` 与 `log π(c|H)`；不要求环境、执行器或 decoder 可微，但训练期间其变化会改变高层动作语义，第一轮应固定。

[TD3 官方算法说明](https://spinningup.openai.com/en/latest/algorithms/td3.html#key-equations) 则通过动作条件 Q 优化 actor。项目若使用 `Q(H,c,F(H,D(H,c)))`，须声明是否保留经过 F 的动作梯度：冻结 F 参数不等于切断对 c 的导数。二者改变的是优化目标/梯度，不能混写为同一版 PPO。

本项目推论：当前采样动作的 `z_phys` 不能直接当普通 PPO 的状态 baseline 输入；`V(H,z_phys(H,c))` 已依赖当前动作。若需要动作条件 control variate，要有修正项，而非直接替换 V；[Q-Prop，式 6–9](https://arxiv.org/pdf/1611.02247) 给出了这种区别。第一版用 `V(H)` 即可，不需要引入 Q-Prop。

## 24 步规划 / 8 步执行如何定义 return

[Diffusion Policy，§II-C](https://arxiv.org/html/2303.04137v4#S2.SS3) 分开 observation、prediction、execution horizon；长计划只执行前缀是既有做法。[RLT，§III、式 3](https://arxiv.org/html/2604.23073v1#S3) 对实际执行 C 步的 chunk 聚合 reward，并以 γ^C bootstrap。

本项目推论：高层 transition 应是 `H_t, c_t, Σ(j=0..k−1)γ^j r_(t+j), H_(t+k), k, done`，通常 k=8；提前终止按实际 k 计。低层 R 每步读实时状态，不能把其叫作 8 步 native-command 开环执行。高层只在重规划时记录一次动作概率；重复计入八次会改变 PPO 目标。

若 R 在前八步也读取整段 τ，后十六步仍可能影响前缀控制，不能简单宣称它们完全无效。但 `F(H,τ_24)` 若假设整条 τ 连续执行，其 24 步后果不等于“执行 8 步后重新采样高层策略”的真实未来。第一轮只对齐实际前缀的后果监督/评价；完整 24 步预测可保留作辅助，须标明延续假设。

## 对 ref1 的最小设计建议

先实现一个明确的高层 RL 合同：绝对 latent actor、固定 decoder/executor、真实前缀 reward、`V(H)`；先检查可执行行为覆盖，再做不带 WM 的 RL baseline。16–64 维、24/8 的具体选择、KL 衰减有效性和 physical token 增益，都是本项目假设，没有被上述来源证明。

特别是 λ_KL→0 只移除行为先验约束，无法突破固定 decoder 的轨迹输出集合；摆脱在线 reference 与突破动作表示瓶颈是两件事。后续加入 physical token 时，固定动作合同和环境预算，以真实策略收益区分 `Q(H,c)` 与带 token 的版本；离线 return 拟合改善本身不能完成仓库 Mission。
