# V1.48 多指几何包围奖励活动

- date: `2026-09-23`
- branch: `agent/v148-grasp-links`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

实验假设、固定源、smoke/正式训练、未见 seed
评估矩阵和停止门槛见同版本 experiment card。
运行前锁定代码 commit、输入/源 SHA、run ID、
GPU5/6、预算和停机条件。工程 smoke 不代表抓取结论。

smoke 从固定 e260 自训练源恢复至 e262，
`qualified_grasp_contact_fraction` 为0.082/0.234，
超过工程门0.05；权重及日志 finite。
正式从原 e260 重新训练至 e300，run manifest
`COMPLETED`，保存 e280/e300；e300 SHA256 见
experiment card。训练过程未使用第二张 GPU。

评测脚本 commit `a120039` 固定 seeds114–118、
每臂每 seed两次、GPU5/6交叉，20个 child 和父
manifest 均 `COMPLETED`。来源、完整 episode 与
关闭提前终止逐项核验；多指配方323/640对
matched Cm-off422/640，seed聚类95%区间为
−19.69至−10.31pp，五 seed 全部退步。
新增输出远低于10GB，总输出仍低于300GB。
此无 Cm 路线按用户对论文目标的提醒停止调参，
转向直接验证 Cm 架构与 PPO 接入的因果作用。
