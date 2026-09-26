# Probe: joint Cm 权重在第二个困难训练 seed 上是否复制？

probe_id: `P-20260923-CM-JOINT-SECOND-SEED`

date: 2026-09-23

branch: `agent/cm-postvalidation-diagnosis`

status: COMPLETED

code commit: `fcade7396c53a966965283b34d24277d6cde1e8c`

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

run_status: `COMPLETED`（训练 1/1、评估 2/2）

conclusion: `UNCLEAR`（有小正差，但未过预定继续门）

| 评估 seed | joint | Cm-off | effect-rank | action-shuffled 参考 |
| ---: | ---: | ---: | ---: | ---: |
| 133 | 43/64 | 38/64 | 29/64 | 52/64 |
| 134 | 36/64 | 33/64 | 20/64 | 46/64 |
| 合计 | **79/128** | **71/128** | **49/128** | **98/128** |

joint−off 为 +8/128（+6.25pp），两次各为正，但未达到预先要求的
至少 +11/128（≥8pp）和总数 82/128。不能因观察到正号后放宽门槛。
上一困难 seed74 joint 为 84/128 vs off 68/128，两个事后选定的
困难 seed 合计 joint 163/256 vs off 139/256；这只是有偏的 Probe
汇总，不是稳定增益的独立验证。两 seed 的 action-shuffled 参考均高于
joint；仍无法证明真实动作对应的 joint 权重有独立作用。

按预定失败分支，不启动 joint 版本的多 seed 正式验证，停止当前
actor 权重族的局部调节。下一路线应换 Cm→PPO 机制。

训练 manifest：`outputs/Dexplore/cm_joint_second_probe_t73/run_manifest.json`；
评估：同目录 `eval_s133_e300_full_cmjoint2/`、
`eval_s134_e300_full_cmjoint2/`。
