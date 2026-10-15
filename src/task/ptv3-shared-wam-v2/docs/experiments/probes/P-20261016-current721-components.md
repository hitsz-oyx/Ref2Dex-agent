---
schema: ref2dex.probe.v2
probe_id: P-20261016-current721-components
experiment_id: P-20261016-current721-components
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: b1c031a9
claim_id: C1
hypothesis_family: HF-current721-components
probe_index_in_family: 1
seed_pool: probe
seeds: [196, 197]
decision_changed_if_positive: retain the most useful current721 blocks in the shared input design
decision_changed_if_negative: treat the current721 gain as distributed or insufficient and avoid hand-picked feature pruning
status: PROMISING
run_id: current721-components-20261016-r1
---

# 721D 当前观测的块级必要性诊断

这是一个低成本 Evidence/Decision Probe，服务于 C1 的输入归因。已有腕部 MLP 的完整
current721 版本在多对象执行上出现对象间差异，本轮用同一个已完成的 `plus_current`
checkpoint 做 leave-one-block-to-mean 消融，先区分“依赖哪一块”与“重训后能否由其他块
补偿”。它不改变四组 shared 训练，也不把后验消融升级为因果结论。

721D 按原生观测合同分成：`body_current[0:252]`、`contact[252:257]`、
`body_reference[257:646]`、`task_object[646:685]`、`interaction_graph[685:721]`。
每次只把该块替换为 TRAIN 统计均值，state87、goal3、标签、模型权重、固定面板和
归一化全部保持不变；另保留 all、全部 current 置均值两个端点。报告 TRAIN/DEV 固定
面板、完整 DEV 以及 startup/later 的 wrist XYZ mean/p95。结果只用于决定后续是否值得
做 matched retraining 或输入压缩，不用于宣称某一字段是唯一根因。

资源边界：GPU1 单次离线推理，输出小于 1GB；GPU0 外部作业和 GPU4/5/6/7 四臂训练不动，
GPU2/3 空闲。非有限输出、源/权重漂移或显存冲突立即停止并保留日志。

## 结果

固定 `plus_current` 6000 步权重的均值替换消融在完整 DEV 上为 40.13 mm。替换整个
`body_reference[257:646]` 后为 60.39 mm，替换 `body_current[0:252]` 后为 98.07 mm；
只替换 body velocity 子块（原生 156:204）就升到 77.70 mm，reference velocity error
（550:598）升到 48.77 mm。contact、reference contact error、task/object 和两块 IG
单独替换均只带来约 1–5 mm 变化。固定权重结果表示模型使用了运动/误差块，不能单独证明
不可替代性。

同架构、同初始化和同采样的 3000 步重训保留了这个块级方向但显示出冗余：完整 current721
为 40.30 mm；去掉整个 body-current 为 43.47 mm，去掉整个 body-reference 为 45.73 mm，
而去掉 task/object 或 IG 分别为 39.52/40.18 mm。只保留 body velocity 或 reference
velocity 均退化到 73.36/80.16 mm，state+goal-only 为 95.98 mm（相同 3000 步预算）。
因此当前最可靠的判断是：真正有用的是一组联合的当前运动与同 tick 参考误差特征，单个
contact/IG 字段不是主要增益来源；不要按固定权重排序贸然裁剪到单一速度块。

产物：`outputs/ptv3-shared-wam-v2/current721-components-20261016-r1/result.json`、
`current721-fine-components-20261016-r1/result.json`、
`current721-retrain-ablation-20261016-r1/result.json` 和
`current721-group-retrain-20261016-r1/result.json`。这是单 seed 离线 Probe，没有闭环或
多 seed Validation；它支持保留完整 721D 作为当前 shared 路线输入，并把“缺乏闭环运动/误差
状态”列为原路线失败的首要可检验解释，不能排除数据准入和 8 拍监督跨度的影响。
