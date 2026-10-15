---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-expanded-training
experiment_id: P-20261015-shadow-expanded-training
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: 98fd8c6c
claim_id: C1
hypothesis_family: HF-shadow-expanded-training
probe_index_in_family: 1
seed_pool: probe
seeds: [214, 215, 216, 218]
decision_changed_if_positive: prioritize native qualification of a sufficiently trained full-data Shadow shared policy
decision_changed_if_negative: inspect fitting and closed-loop distribution shift rather than assume data quantity alone
status: RUNNING
run_id: shadow-expanded-20261015-r1
---

# Shadow shared 全量数据双卡扩大训练

Class: Decision，服务独立执行底座。用户主动要求停止Inspire shared_tau，GPU4/5改跑Shadow
扩大训练，并允许其余空卡用于Inspire抓取评估。之前Shadow仅1k×64，不足以评价全量收敛；
当前起步采样版保持4/16、teacher7/16，但起步DEV动作拟合仍差，结果UNCLEAR。
检验明显增加优化曝光能否同时改善heldout拟合与固定native面板，不把跨手差异当数据量因果。

## 固定实现与读出

从`shadow-startup-sampling-20261015-r1/train-startup/step1000.pt`继续模型和完整AdamW状态。
不使用teacher actor初始化，不做GTtau warmup，不新增数据或标签。原全量TRAIN1567物体/
10118轨迹/1536897窗，原起步50%+序列均匀50%混合不变；全部TRAIN-only normalizer保持原值，
forecast父统计保持remote80k。保留同300D当前状态+shared→24D原生命令及hand/scene forecast。
仍为geometry/shared/actor LR5e-6/1e-5/5e-4，权重1、梯度裁剪1、AdamW wd.01、AMPbf16/TF32。

两卡DDP，每卡256/global512，从1000继续到总20000，增加19000×512个抽样。
改变batch/world-size，重建按step/rank确定的新采样流；不声称逐样本复现单卡历史。
每rank2loader workers；checkpoint每1000步完整保存（模型/optimizer/两rank RNG/采样计数）。
启动前同batch双卡4-update smoke从原1000独立启动，正式不继承smoke更新。
固定原common DEV64/seed215及startup DEV64/seed218，后者仅报告，不用于选点。
best按common DEV normalized action L1；另存forecast_best与各阶段，防止动作/forecast取舍丢失。

后续闭环先用2000阶段在原16起点/seed216复核，再根据实际拟合/吞吐决定5000/10000阶段
是否值得评估；评估使用空卡，不阻塞主训练，不把单阶段/单seed成功当正式结论。
原1k完整结果是基线，不覆盖它。没有把当前跨手对比重标为matched control。

## Decision Note / 资源

新用户指令明确扩大授权：终止supervisor1922941拥有的Inspire shared_tau组，保留最近完整7k
及历史阶段；GPU4/5仅新Shadow，两组Inspire0..3继续。GPU6/7允许并行做当前抓取评估。
本轮两卡<=12h、总输出32GB、磁盘20GiB底线。预期实际吞吐以首40更新与GPU采样为准，
启动后向用户报告显存/利用率/ETA；若预测超12h，不静默延长预算。
非有限、输入/normalizer不匹配、外来GPU进程、30分钟无进展或资源超界只终止自己的进程组。

## Limitations / future evidence

续训同时改变训练曝光和batch，非单因素数据规模对照；normalizer来自早期匹配小TRAIN统计。
DEV64对总体泛化代表有限；独立seed、大DEV面板、完整teacher-state-only基线和正式成功率验证
待有可决策信号后再做。当前Probe仅能PROMISING/UNPROMISING/UNCLEAR。


## 实际启动 / 2026-10-15

固定detached代码98fd8c6c，tmp/shadow-expanded-code-20261015-r1。双卡global512、4更新
工程smoke r2 COMPLETED，完整step1004 CPU审计PASS：optimizer step1004、两rank RNG、
模型有限、起点DEV action与原读出差5.48e-7。原normalizer/模型/优化器严格恢复。
独立review指出的elastic rank进程组归属误判已改为父子树；best含1k起点评估，存档保留
sampler/comparison与rank RNG以支持后续恢复。DDP沿用Inspire的rank-local masked loss平均，
scene motion权重分母每rank独立，不宣称与单卡pooled512比值严格相等。

正式不继承smoke更新，重新从原startup1000启动，supervisor2250275/GPU4,5，
输出outputs/shadow-finetune/shadow-expanded-20261015-r1。启动baseline两个固定DEV面板已记录。
当前RUNNING，首步1001有限；吞吐/ETA待稳定若干更新，不用首步启动开销外推。


稳定启动读出：1140/20000，1.672s/update、预计余8.76h（不含将来的保存/评估开销）。
20秒六次采样GPU4/5平均利用率95.83%/99.67%，nvidia显存10379/10327MiB，
峰值allocated8847.9MiB。资源合理，仍在12h预算内，不据短期TRAIN loss判断拟合收益。
报告startup-resource-readout.json。两rank完整optimizer起点1000/原三LR已审计PASS。
固定step2000单次16env/seed216闭环watcher2254836已启动，等待checkpoint和GPU6空闲；
最多等1h+评估20min，不阻塞训练，不自动扩展到其他阶段。程序为evaluate_milestone.py。
21项相关合同测试通过，workflow verify PASS；当前仅启动和工程验证，无充分训练后的科学结论。
