---
schema: ref2dex.probe.v2
probe_id: P-20261014-ptv3-shared-v2-pretrain
experiment_id: P-20261014-ptv3-shared-v2-pretrain
date: 2026-10-14
task: ptv3-shared-wam-v2
branch: main
git_commit: da694f1f
claim_id: C1
hypothesis_family: HF-ptv3-shared-v2-pretrain
probe_index_in_family: 1
seed_pool: probe
seeds: [136]
decision_changed_if_positive: continue bounded v2 pretraining if physical hand and scene errors improve on the fixed three-domain dev panel
decision_changed_if_negative: audit input routes and normalized target semantics before increasing compute or rejecting v2
status: PLANNED
run_id: pretrain-main3-probe-20261014-r1
---

# 联合轻量PTv3与大shared的horizon标准化预训练

Result: PENDING. CPU合同测试13项通过，待真实GPU工程检查和1000步Probe。
Decision: 用户已经同意新Task与双路hand/统一horizon方案；按自主探索规则先做最小可判别训练。

Class: Decision. 服务于三域交互表示与最终机器人适配；问题是减少几何参数、让hand进入
PTv3并使用统一horizon统计后，该v2组合是否能稳定学习物理手/物体预测。
1000步最便宜地区分可训练的候选与先修实现/优化的候选，不直接进行长训练或正式消融。

## Decision Note

- 当前旧660微调在GPU5/7继续，不修改其模型/训练源码。v2在当前main独立目录开发；
  用户没有授权新分支，故不新建分支。用户原会话批准的模型方向不改变MISSION claim。
- 输入以固定物理尺度处理；输出统计仅来自TRAIN共享4096窗。手编码前后双路；
  22手点独立回读。参数从头初始化，不强行迁移维度不同的旧PTv3。
- 先统计CPU数组（无神经计算），真实GPU四步接线/负载/恢复检查；通过后2卡4/6，
  每卡64/global128、1000次optimizer更新，最多2小时、单run8GB、共享总量400GB。
  预计10–40分钟，以实测吞吐更新。与旧训练合计4卡，不影响其他用户GPU0–3。
- 任一非有限量、梯度合同失败、外部GPU进程出现、源码变化、资源超限立即停止本run。
  不杀未知进程，不覆盖旧权重。只有实施结果具备改善信号再考虑更长预算。
- 成功条件为三域hand_all24与scene_all24物理EPE相对step0多数域改善，并且没有
  未解释的归一化/有效性错误。此门用于选择是否继续，不是正式显著性或因果归因。
  若某域恶化先核对幅度和mask，不把组合短训负结果升级为核心假设失败。

## 固定协议

见[架构](../../user/架构.md)与[配置](../../../configs/pretrain_probe.json)。
数据：既有pointworld-main3-sharedstats数据，无新增下载；原完整序列划分和采样比例。
TRAIN统计run `horizon-stats-main3-20261014-r1`，seed18（debug统计抽样，不是训练重复），
按mix sampler采4096窗，场景/手分别24×3 mean/std，std floor1mm，保存train_indices。
输出：`outputs/ptv3-shared-wam-v2/`，数据与checkpoint不复制到文档目录。
训练seed136、rank初始化136/137；dev seed138、96窗、每域32窗。rank不是科学重复。
AdamW，geometry/shared均1e-4、weight_decay=.01、clip1；不增加额外重建损失。
bf16、TF32、shared激活重计算、有界4workers/rank、prefetch2、每worker/source缓存128。
每250步验证，保存所有best与500步恢复点并保护best/latest，最多保留best+recent与最终。
metrics复用已审查的物理EPE与goal-only线性插值分组；normalized objective不与v1直接比较。

## 工程与运行记录

- 初始实现提交da694f1f；13项CPU测试覆盖mask/NaN、horizon24×3 roundtrip、同voxel
  密度不冲淡手分支、22语义槽位、goal到双头、梯度隔离与v1 checkpoint拒绝。
- 真实GPU工程检查、参数实测与吞吐在执行后补充。工程run不作为学习证据。
- 实际run代码提交、worker PID、输入、统计、配置与初始化由manifest.json记录。

## Limitations / future evidence

单seed、1000更新只用于可训练性与继续投入决策，不能形成机器人抓取/Cm收益结论。
三项结构改动与归一化一起改变，没有匹配消融，不能把改善归因某个模块。
统计来自4096个TRAIN抽样窗，并非全数据精确统计；缺少独立采样误差评估。
后续正式比较需要固定物理指标、相同数据预算/初始化条件、多seed以及robot/native验证。
旧Task的微调继续使用旧统计与checkpoint；v2目前没有robot迁移执行入口。
