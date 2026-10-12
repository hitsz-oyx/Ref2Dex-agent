# SelfWAM：执行动作条件与信息隔离核验

日期：2026-10-12。来源是 [arXiv v1](https://arxiv.org/html/2608.00725v1)、[作者项目页](https://selfwam.github.io/) 与 [作者链接的仓库](https://github.com/SelfWAM/SelfWAM)。未运行代码。

## 核实的方法及限制

[论文 §3](https://arxiv.org/html/2608.00725v1#S3) 对 demonstrated action 建立 clean copy，使用同一个 action tokenizer/blocks、干净噪声时刻及对应的时间位置；future visual queries 能看到 clean action，noisy action queries 不能看到 clean action 或 future visual tokens。clean action queries 自身也不能看到 future visual tokens，避免间接泄漏。动作部署时去掉未来视频及 clean-action 条件路径；可选 rollout 可代入 policy sample 或 candidate action。

未来 robot self-mask 是同一视频 backbone 的另一个输出域，mask 只作训练目标。mask 样本不再次计 action loss。[论文 §3](https://arxiv.org/html/2608.00725v1#S3)

[论文 Table 4](https://arxiv.org/html/2608.00725v1#S4) 的 policy 平均成功率：FastWAM 91.84，clean-action RGB-only 90.80，完整 SelfWAM 92.62。因此信息流更清楚不等于仅加 action conditioning 就提高 policy。论文未采用 goal endpoint consistency loss 来反向优化 actor，也未证明给定几何目标可执行。

[作者项目页](https://selfwam.github.io/) 展示固定 observation 下更换/缩放 action 的未来响应；这是 action sensitivity 的证据。它与我们的 goal-to-action sensitivity 是不同干预链。训练初始化 Wan2.2-5B，使用 64 A800、global batch 1024、5 epochs，不能当作从头小模型短训的直接成功保证。[论文配置](https://arxiv.org/html/2608.00725v1#S4.SS1)

作者仓库本次可见 release 分支仅 README、docs、LICENSE，尚未核验可运行源码。

## 本仓库建议（推断）

现有 `W(H, executed trajectory)` 与 actor 不看 executed-action 标签的合同可继续保留；共享当前 scene 特征本身不构成标签泄漏。接入 goal→actor→world loss 前，还需固定 H 更换 executed action，检查物体预测是否确实依赖 action；目标物体作用、接触和不确定性不由机器人 self-mask 证据自动解决。

冻结 W 能阻止它与 actor 联合篡改预测映射，但不能阻止 actor 利用 W 的已有误差。闭环一致性应待世界模型的动作敏感性和相应物体预测误差得到针对性诊断后再做有界 Probe，不据该论文直接宣称 world 是物理约束或 goal loss 保证使用 goal。
