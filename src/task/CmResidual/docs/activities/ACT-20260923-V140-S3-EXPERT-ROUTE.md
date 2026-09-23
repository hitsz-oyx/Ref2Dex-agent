# V1.40 自训练 s3 checkpoint 互补性与固定整段路由

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

四枚冻结自训练 PPO checkpoint 的 SHA 与发现集/停损阈值
见 experiment card。先补齐 seeds82–84 的 source、standard、
back240 严格完整轨迹结果，back260 已在 V1.39 完成；
然后计算同环境 oracle 并集。只有每 seed 均≥58/64，
才实现固定开始帧路由并在新 seeds85–89 复评。
