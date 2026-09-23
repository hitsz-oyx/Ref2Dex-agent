# Probe: joint Cm 权重在第二个困难训练 seed 上是否复制？

probe_id: `P-20260923-CM-JOINT-SECOND-SEED`

date: 2026-09-23

branch: `agent/cm-postvalidation-diagnosis`

status: PLANNED

## Question and decision

joint 在 seed74 的两次评估相对 Cm-off +12.5pp，通过小 Probe 门。
但 seed74 是已知困难 seed，且 action-shuffled 更高。本 Probe 只回答：
原 joint 配方是否在另一此前 effect-rank 失败的训练 seed73 也有
明显正信号？若有，设计新训练 seed 的 joint/off/full-weight-permutation
正式验证；若无，停止将 joint 视作稳定候选。

## Minimal protocol and fixed gate

同一自训练 s3 e260 checkpoint、CmLite V1.37、corrected_r2 输入；
只训练 seed73 的原 `joint` mode 至 e300，其他训练超参完全复制
`VAL-20260923-CM-EFFECT-PPO`。严格评估 seed133/134、各 64env
首个完整 episode。既有同训练 seed73 的对照为 off 38/64、33/64，
effect-rank 29/64、20/64，action-shuffled 52/64、46/64。

推进门：joint 两次合计至少 82/128（比 off 71/128 高≥11/128，
即≥8pp），且每个评估 seed 都高于 off。没达到则不启动 joint
正式验证。此 seed 和评估对照已知，结果只服务继续/停损，非独立验证。

## Budget and stop

1 GPU，预计 <30min、新产物 <2GB。代码 commit、输入 SHA、GPU、
唯一 run_id 固定后启动；代码/输入漂移、GPU 冲突或合同失败则停止。

## Result

待运行。
