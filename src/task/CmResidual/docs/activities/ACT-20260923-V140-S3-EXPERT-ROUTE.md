# V1.40 自训练 s3 checkpoint 互补性与固定整段路由

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- evaluation code commit: `2c03929`
- run_status: `RUNNING`
- official actor checkpoint: `null`

四枚冻结自训练 PPO checkpoint 的 SHA 与发现集/停损阈值
见 experiment card。先补齐 seeds82–84 的 source、standard、
back240 严格完整轨迹结果，back260 已在 V1.39 完成；
然后计算同环境 oracle 并集。只有每 seed 均≥58/64，
才实现固定开始帧路由并在新 seeds85–89 复评。

已启动发现集缺失评估第一组：GPU5 的 source e180 seed82，
GPU6 的 standard e260 seed82；均为64环境首个完整 episode，
提前终止关闭。两 run manifest 初态 `STARTED`，记录
冻结 checkpoint SHA、输入与精确命令。之后按相同口径补齐
seeds83/84 及 back240，不边看结果边增删专家。

发现集缺失评估均已 `COMPLETED`。四专家同环境 oracle 并集
按 seed81/82/83/84 分别为 **62/59/64/63**，全部达到
预设的58/64开发门槛；这仍是不可部署的上界。
用 `fit_s3_episode_router.py` 按冻结目标拟合出的4段路由：
0–15 `back260`，16–29 `source`，30–34 `back240`，
35–40 `back260`。发现集实际路由成功数为
**53/54/53/59**，总目标分数213（219成功减6复杂度罚项）。
路由文件在 `outputs/CmResidual/agent_v140_s3_router_fit/route.json`，
记录所有发现输入结果 SHA；路由不使用 seed、接触或未来结果。
下一步冻结文件，在全新 seeds85–89 上评估。发现集表现
低于90%目标，故对 heldout 不作乐观预期。
