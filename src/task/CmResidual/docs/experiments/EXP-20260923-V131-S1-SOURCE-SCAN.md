# V1.31: 自训练 s1 源 checkpoint 的 s3 零样本迁移扫描

- experiment_id: `EXP-20260923-V131-S1-SOURCE-SCAN`
- branch: `agent/multitrajectory-v129`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（扫描网格中不存在优于 e140 的直接迁移源）
- official actor checkpoint used: `no`

## 假设与冻结设计

V1.30 从自训练 s1 epoch 140 源迁移微调 s3：Cm-off 达到未见
seeds 71–73 的 83/192，但仍未稳定；同样训练预算加入当前 CmLite
奖励则为 30/192。下一步先检查源策略选择是否成为迁移瓶颈，避免
用一次任意的 e140 源作为唯一结论。

固定源训练 `agent_v125_norm_cmlite_anneal_s45_e200`，其 run manifest
为 `COMPLETED`，没有官方 actor。只评估该**自训练** run 的 epochs
100、120、130、150、160、170、180、200，另以已经完成的 e140
2/64 为基线。冻结 s3 `corrected_manifest_r2.json` 与对应 motion root，
每个 checkpoint 在 seed 71、64 env、完整轨迹且提前终止关闭的首个
episode 上评估。该阶段无训练或策略选择反馈。根据严格成功率排序，
若最佳值高于 e140 的 2/64，再取最佳的至多两枚 checkpoint 在未见
seeds 72、73 上复评；否则停止，不启动以此为源的新训练。

首轮每枚 checkpoint 最多一次评估；GPU 5/6 各一进程，同时最多两卡；
预计不超过 10 分钟，若发现外部占卡、输入漂移或单次异常则停。
指标为连续至少 5 步手物接触且物体抬升至少 3 cm 的 episode
比例，辅以平均最大接触抬升。seed 71 用于挑选，seed 72、73 才是
选中 checkpoint 的独立复核；不能把 seed 71 扫描最大值视为
泛化估计。此实验只考察源选择，不证明 Cm 有用。

## 结果与停损

同一 s3 seed 71 完整轨迹门禁：

| 自训练 s1 checkpoint epoch | 严格成功 |
| ---: | ---: |
| 100 | 0/64 |
| 120 | 0/64 |
| 130 | 0/64 |
| 140（V1.29 已评估基线） | 2/64 |
| 150 | 0/64 |
| 160 | 1/64 |
| 170 | 0/64 |
| 180 | 0/64 |
| 200 | 0/64 |

预注册的八个新增 run manifest 均为 `COMPLETED`，其 checkpoint SHA、
输入 SHA、评估代码提交和结果各在源 run 的
`eval_s71_e{epoch}_full_s3_source_scan/` 下。没有候选超过 e140，
故不执行 seed 72/73 的下一阶段，也不以这些候选启动新微调。
这只排除了当前 s1 run 的该 checkpoint 网格，不能推断所有可能的
自训练源策略均无迁移性。
