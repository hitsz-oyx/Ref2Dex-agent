# V1.44 同策略重复物理评测

- date: `2026-09-23`
- branch: `agent/v144-replay-audit`
- run_status: `PLANNED`
- official actor checkpoint: `null`

按 experiment card 固定 seed98、GPU5、V1.43
评测器与同一 SHA 的路由/专家；先 off 再 always，
各一次。只比较重现性，不根据结果改门槛。
