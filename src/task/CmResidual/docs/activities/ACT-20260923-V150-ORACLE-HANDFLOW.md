# V1.50 原版 Cmv2 手流误差来源活动

- date: `2026-09-23`
- branch: `agent/v150-oracle-handflow`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

固定输入、三个手流条件、oracle 未来状态禁入在线
策略的边界、smoke/正式运行、GPU/预算与科研门槛
见同版本 experiment card。

smoke 与正式 run 均由代码 commit `fd89143`
在空闲 GPU5 完成；manifest 锁定原版 checkpoint、
转移 SHA、相同抽样、microbatch2/8、oracle
未来信息禁入在线策略。模型与输入 finite，
未修改外部项目、数据或策略。

正式256样本的接触手点流名义均值52.8/57.1mm，
真实8.2/12.0mm；原版 Cmv2 平移 EPE 从
39.05/52.74mm 降到9.02/14.60mm，但零物体
位移只有5.36/10.35mm。强门槛失败；详见
experiment card 和 `audit.json`。运行仅是离线
误差归因，不是 Cm 参与 PPO 的正面证据。
