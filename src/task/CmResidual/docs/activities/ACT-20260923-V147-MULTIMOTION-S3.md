# V1.47 s1+s3 多轨迹训练活动

- date: `2026-09-23`
- branch: `agent/v147-multimotion-s3`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

实验设计、固定输入、门槛与限制见同版本 experiment card。
本次在工作区内使用 s1/s3 张量目录软链接；外部项目只读。
训练开始前须记录 git commit、manifest SHA、run ID、
GPU、预算、停止条件；smoke 只验证接线，不能证明抓取效果。

固定输入合同 SHA256 `9a101e68df1b7f60eb3109daad86c1baee565e19192799e04d05c1fb15604f8c`，
s3 输入 manifest SHA256 `2878bd20d1dd849f6844883c832a3777602d2d2bc73f7581b3bc23d30725f038`，
训练代码 commit `70b082f`。GPU5/6 同时最多各一进程，
输出新增约1GB，整体远低于300GB；没有外部项目写入。

两臂 e262 smoke `COMPLETED`，确认同源 e260 恢复、
`1e-5` 学习率、mixed 2条motion、接触课程与 finite
训练 reward，均产出 e262 checkpoint。正式两臂从
原 e260 而非 smoke 权重独立重启，e280/e300/e320
均保存，manifest `COMPLETED`。最终 e320 权重 SHA
与严格评测见 experiment card。

评测脚本 commit `b9a514d` 固定 seeds109–113、
每臂每 seed两次、GPU5/6交叉；20个 child manifest
及父 manifest 全部 `COMPLETED`，无失败。逐项
核验后 mixed 225/640、control 200/640、差+3.9pp，
seed聚类95%区间跨0且 seed113 为负；两个预注册
门槛失败。按照停损规则，不再对这一配方做 s1
保留评估或基于这些 heldout seeds 调参。
