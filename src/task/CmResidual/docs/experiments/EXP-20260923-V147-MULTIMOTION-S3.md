# V1.47: s1+s3 混合参考轨迹继续训练对 s3 稳定抓取的影响

- experiment_id: `EXP-20260923-V147-MULTIMOTION-S3`
- branch: `agent/v147-multimotion-s3`
- run_status: `COMPLETED`
- conclusion: `REFUTED`（预注册相对改善及稳定抓取门槛）
- official actor checkpoint used: `no`

## 冻结假设与输入

V1.45 的 s3 专家整段路由重复评估达 499/640=78.0%，
但十次 run 最高仅 53/64，未到每次 ≥58/64 的稳定门槛。
V1.46 的 CmLite 单步预测进度奖励比 matched Cm-off 差 21.4pp，
本实验暂不加入 Cm，只问第二条 GRAB 重定向轨迹作为 PPO
训练分布是否改善 s3 泛化。它不能证明 Cm 有用。

两臂都从**自训练** s3 `back260` e260 checkpoint
SHA256 `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`
恢复。`s3_only` 使用原 s3 corrected_r2 tensor SHA256
`64d18ee69ea3fcb7ed76300aa2f5004383b7c01b29a30bc533f6960e3db5be1c`；
`s1_s3` 使用固定目录中的两个**目录软链接**，另加 s1
coordfix_v4 tensor SHA256
`305dfd16d9bac0e93a95de5b6fa8b9e721d1edfee6946e49100fe231760a205b`。
输入合同见 `configs/v147_s1_s3_motion_manifest.json`；
外部 `/home2/wyy/oyx_ws/Ref2Dex` 只读，不复制也不改动。
DExplore 将64个环境按 motion ID 轮转分配；混合臂每条32环境。
s1 参考右手接触标签全为 −1，s3 从帧50有正标签；
这使混合臂的接触模仿奖励语义并不完全一致，是预先记录
的解释限制，不得事后忽略。

两臂仅 `motion_root` 不同，其余固定：Cm-off、seed70、
64环境、horizon32、minibatch256、学习率1e-5、从 e260
续训至**固定 e320**、每20 epoch保存；几何接近/持物抬升/
抬升进展奖励2/10/5；接触重置前后各3帧、
contact/lift fraction 0.5/0.25、backtrack180–220。
混合臂一半训练环境来自 s1，故不能把差异纯归因于
“增加样本”而排除 s3 有效训练量减半。

先各跑到 e262 的2-epoch工程 smoke：源权重和输入 SHA
一致、两种 motion 均加载、checkpoint/奖励有限、
课程重置无错误。任一失败则停止正式训练，修接线另立
run ID；smoke 权重不用于正式实验。正式训练完成后
固定 e320，不按评测结果选 epoch。

在全新 s3 seeds109–113 上两臂各 seed 重复2次，
每次64个完整首 episode、关闭提前终止，重复轮次交换
GPU5/6，不把 env_id 当可重复物理状态。主门槛：
混合臂总成功率比 s3_only 高≥8pp、五个 seed 的两次
均值均正、按 seed 聚类10,000次 bootstrap 的双侧95%
区间下界>0。稳定门槛另要求混合臂十次均≥58/64。
若主门槛失败，停止此配方，不在 heldout seeds 上调参。
若主门槛通过，再用新 seeds114–118 对 s1 原稳定基线
和混合臂做独立检查；这不是 s3 主结论的组成部分。

GPU5/6最多并行两卡，任何卡使用量>1GiB的外部进程
出现则不启动下一组；不停止他人进程。预计训练和评估
共<1小时，新增产物<20GB，总产物<300GB。
固定输入/代码/源权重漂移、非有限数值或资源冲突则停机。

## 固定 e320 未见 seed 结果

两臂 e262 smoke 与正式 e320 训练均 `COMPLETED`；
正式训练固定代码 commit `70b082f`，mixed e320 SHA256
`6b25eaff80797133b80c1ebc65c5da9f54d082a0ffec4a0147aa899ef5f26be6`，
control e320 SHA256
`2ec9e87ad2f534643b8418d6bd250af2a7564429c353008fa738040c9f0ee360`。
两臂均由同一 e260 自训练权重重新启动，不使用 smoke 权重。
混合臂确实加载两个 motion，`first_contact_frames=[60,50]`；
单轨迹臂为 `[50]`。以下全部评估统一使用 s3 tensor，
每 run 64环境首个完整 episode，关闭提前终止，
没有 Cm 推理模型：

| seed | mixed r0/r1 | s3_only r0/r1 | 差/128 |
| ---: | ---: | ---: | ---: |
| 109 | 23 / 24 | 20 / 23 | +4 |
| 110 | 23 / 28 | 22 / 23 | +6 |
| 111 | 20 / 26 | 15 / 15 | +16 |
| 112 | 21 / 22 | 20 / 18 | +5 |
| 113 | 14 / 24 | 20 / 24 | −6 |
| 合计 | **225/640 (35.2%)** | **200/640 (31.3%)** | **+25/640 (+3.9pp)** |

seed 聚类、固定 RNG147 的10,000次 bootstrap 双侧95%区间
为 **−1.09至+9.06pp**；含0，且 seed113 为负、
总体不足预注册+8pp。十次 mixed 没有一次达到58/64，
所以相对增益与稳定门槛均失败。按冻结规则不在
seeds114–118做 s1 保留评估，也不在本配方上用这些
heldout seeds 调参。mixed 平均最大接触抬升0.116m，
control0.081m；平均接触占比0.354对0.351。
两臂整 run 含初始化中位67/66秒，不是纯推理延迟。

此结果**不支持**“再加入这条 s1 轨迹并继续60 epoch
可使 s3 稳定抓取”。两臂都比先前 e260 单策略
V1.45 的67.3%低得多，但那些是不同 seed；
不能仅据此归因于训练轮数造成灾难性遗忘。
另一个预先记录的限制是 s1 参考接触标签全 −1，
不宜把负结果泛化为所有质量合格的多轨迹数据。
本实验没有 Cm，不能支持其有用或无用的结论。

运行证据位于 `outputs/Dexplore/agent_v147_{s3only,s1s3}_s70_e320/`
和 `outputs/CmResidual/agent_v147_eval_matrix/`，
后者的 `analysis.json` 逐项核验来源、完整 episode
及固定矩阵。
