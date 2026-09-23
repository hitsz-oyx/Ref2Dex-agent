# V1.29 s3 轨迹迁移活动记录

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- code commit: `557702b`
- run_status: `COMPLETED`

保留工作区原有的 `AGENTS.md` 修改、`data`/`dataset` 链接、V1.21c
未跟踪文件及 DExplore 生成资产。未改动工作区外的 Ref2Dex 项目，也未
停止其他用户进程。训练和评估最多同时使用 GPU 5、6；当前产物约
34 GiB，低于 300 GiB 上限。

已完成：

- `agent_v129_s3_cmlite_s66_e200`：旧 CmLite、scratch PPO、GPU 5、
  seed 66、200 epoch。`run_manifest.json` 已记录终态；独立 seed 67
  的五个检查时点在默认提前终止评估下均为 `0/64`；与 V1.28
  路由结果比较前需使用无提前终止口径复核。
- `eval_s67_e050/e100/e130/e170/e200`、`refaction_s67_lead1`：
  产物位于上述训练目录，含逐 episode JSON 和日志。
- `outputs/CmLite/V1.29/s1_s3_abs_train_s68_val_s69` 与
  `s1_s3_rel_train_s68_val_s69`：匹配数据、30 epoch、GPU 6，
  训练终态 `COMPLETED`，日志分别是
  `outputs/CmLite/V1.29_abs_train.log`、`V1.29_rel_train.log`。
- `agent_v129_s3_relcm_smoke_s70_e3` 与
  `agent_v129_s3_cmoff_smoke_s70_e3`：GPU 5/6、3 epoch，均完成。
- `python -m pytest -q src/task/CmResidual/tests/test_cmlite.py`：`5 passed`。

已完成：`agent_v129_s3_relcm_s70_e200` 与
`agent_v129_s3_cmoff_s70_e200`。两者工作版本 V1.29、代码 commit
`557702bd41738fee24006d949575bb136df99711`、输入 manifest
`outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json`、
200 epoch 上限、每 10 epoch 保存，均正常完成。seed 71 的 epoch
100/140/180/200 完整轨迹评估均为 0/64，触发预注册停损；逐 checkpoint
manifest 位于各训练目录的 `eval_s71_e{100,140,180,200}_full`。旧 s3
run 的 `eval_s71_e100_full` 和 `eval_s71_e200_full` 也均为 0/64。

正式评估使用 `tools/eval_dexplore_full_episode_grid.py`，逐 checkpoint
记录 SHA256、运行命令、GPU、代码提交与结果，并强制
`early_termination_disabled=true`。在旧 s3 run 上的双 checkpoint
dry-run 与正式运行均通过。

s1 自训练 PPO epoch 140 迁移 s3 的运行在
`outputs/Dexplore/agent_v125_norm_cmlite_anneal_s45_e200/eval_s71_e140_full_s3_transfer`
取得 2/64；s1 DAgger BC 迁移在
`outputs/CmResidual/agent_v129_s1bc_to_s3_s71_n64_r2` 为 0/64。
首次 BC 调用因工作目录错误失败，证据保留在无 `_r2` 的运行目录；
修正工作目录后未改变数据或策略。

为下一轮 s1 scratch checkpoint 的匹配微调扩展 Cm-off 的 SHA 锁定恢复，
对应测试 `14 passed`，尚未把 V1.29 的 0/64 解释为所有迁移方法失败。
