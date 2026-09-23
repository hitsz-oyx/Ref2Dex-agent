# V1.29 s3 轨迹迁移活动记录

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- code commit: `557702b`
- run_status: `RUNNING`

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

运行中：`agent_v129_s3_relcm_s70_e200` 与
`agent_v129_s3_cmoff_s70_e200`。两者工作版本 V1.29、代码 commit
`557702bd41738fee24006d949575bb136df99711`、输入 manifest
`outputs/CmResidual/agent_v129_s3_coordfix/corrected_manifest_r2.json`、
200 epoch 上限、每 10 epoch 保存。终态记录最后 epoch、checkpoint、
独立严格评估结果及完成/失败原因；科学解释更新对应 experiment card。
