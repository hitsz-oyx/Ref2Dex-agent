# V1.31 自训练 s1 源 checkpoint 的 s3 零样本迁移扫描

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- evaluation code commit: `fbf717b`
- run_status: `COMPLETED`
- official actor checkpoint: `null`

输入是 V1.29 s3 重建数据 `corrected_manifest_r2.json`；使用现有
`eval_dexplore_full_episode_grid.py` 的 V1.29 严格评估协议与
`--motion-root-override`、`--input-manifest-override`、`--tag s3_source_scan`。
源为本地自训练 s1 run `agent_v125_norm_cmlite_anneal_s45_e200`，
只读其现有 checkpoint；独立运行分别记录 checkpoint SHA256、
代码提交和 GPU。命令、run_id 与终态写在各评估目录的
`run_manifest.json`。详细假设与停损见对应 experiment card。

已在 GPU 5 依次启动 seed 71 的 e100/e120/e130/e150，在 GPU 6
依次启动 e160/e170/e180/e200；run_id 采用
`eval_s71_e{epoch}_full_s3_source_scan`。两进程各运行一枚 checkpoint
一次，预定首轮最多八次评估、约十分钟。各评估目录创建时标记
`STARTED`，完成后由评估器写入 `COMPLETED` 和结果。

八枚 checkpoint 的 seed 71 完整轨迹评估全部正常完成；最高为 e160
的 1/64，仍低于原 e140 的 2/64，触发停损。评估终态、严格指标
与 checkpoint SHA 位于对应 `run_manifest.json`，详细结果见
experiment card。GPU 5/6 已释放；本轮没有训练、没有删除产物。
