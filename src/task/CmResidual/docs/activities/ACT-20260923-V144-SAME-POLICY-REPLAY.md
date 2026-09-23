# V1.44 同策略重复物理评测

- date: `2026-09-23`
- branch: `agent/v144-replay-audit`
- run_status: `COMPLETED`
- evaluation code commit: `250efd8`
- official actor checkpoint: `null`

按 experiment card 固定 seed98、GPU5、V1.43
评测器与同一 SHA 的路由/专家；先 off 再 always，
各一次。只比较重现性，不根据结果改门槛。

GPU5上顺序重跑 off/always，run manifest 均为
`COMPLETED`。和各自 V1.43 原 run 比较，起点
均64/64一致，但首次接触前同进度仅37/35，
严格同状态均0/64；成功数 off 49→56、
always 45→52，逐环境结果翻转15/23。
达到预注册的“评测配置不重现”判定，停止
跨 run env_id 级因果解释。下一轮需要对
更大的策略差异做多 seed 重复分布比较。
