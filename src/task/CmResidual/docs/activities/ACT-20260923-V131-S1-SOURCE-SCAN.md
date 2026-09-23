# V1.31 自训练 s1 源 checkpoint 的 s3 零样本迁移扫描

- date: `2026-09-23`
- branch: `agent/multitrajectory-v129`
- run_status: `PLANNED`
- official actor checkpoint: `null`

输入是 V1.29 s3 重建数据 `corrected_manifest_r2.json`；使用现有
`eval_dexplore_full_episode_grid.py` 的 V1.29 严格评估协议与
`--motion-root-override`、`--input-manifest-override`、`--tag s3_source_scan`。
源为本地自训练 s1 run `agent_v125_norm_cmlite_anneal_s45_e200`，
只读其现有 checkpoint；独立运行分别记录 checkpoint SHA256、
代码提交和 GPU。命令、run_id 与终态写在各评估目录的
`run_manifest.json`。详细假设与停损见对应 experiment card。
