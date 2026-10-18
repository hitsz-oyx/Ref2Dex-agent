---
schema: ref2dex.probe.v2
probe_id: P-20261018-scheduled-wam-bc
experiment_id: P-20261018-scheduled-wam-bc
date: 2026-10-18
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 1
seed_pool: debug
seeds: [0]
decision_changed_if_positive: qualify actual scheduled-WAM BC as a candidate for native execution smoke before PPO
decision_changed_if_negative: inspect conditioning distribution and supervision before scaling policy training
status: COMPLETED
run_id: scheduled-wam-bc-20261018-r1
---

# 调度WAM条件下的UniDex BC初始资格

Class: Decision。旧GT-τ BC在一个Camera实测状态、当前WAM预测条件下有6/24通道
超出动作范围；该单点不是机器人失败证据。当前最便宜的判别是固定WAM，检查实际
部署条件下BC能否在独立raw episode的动作标签上拟合，避免直接扩大失配的PPO。

固定修复版v4 step12000；九源预训练不改。缓存
`outputs/shadow-finetune/policy-data-full-20261015-r1`的Camera TRAIN记录7222/7223
拟合，7319/7320作为内部开发；两组raw episode分别为episode00/01。对象相同，因此
不声称对象泛化，内部开发也不是全局DEV/TEST。移植小包后ID映射0/1→train、2/3→dev。
每记录按时间均匀选32个合法当前状态，合计128行；调度器每60Hz实测帧更新，H4来自
过去7帧的0/2/4/6，路径来自初态与任务goal，不来自未来轨迹。检查canonical action与
下一拍观测previous-action一致。只有TRAIN拟合归一化。

403→256→256→24 ELU网络从随机初始化做1000步BC，Adam1e-3，seed0，固定末步
checkpoint，不按开发曲线选择。报告TRAIN/内部dev L1、dev动作裁剪比例及条件来源。
此阶段只为接口与可拟合性资格，不据此宣称抓取收益；若dev裁剪仍>=10%，或dev L1
不优于TRAIN动作均值常量，先诊断而不扩大PPO。即使通过，仍须真实闭环smoke。

远端本项目GPU0短时共享，graspenv Python3.8.20 / Torch2.0.1+cu118；预算600秒、
输出<=1GiB、空盘>=20GiB。不修改/停止主WAM训练。产物在既有NAS的
`tmp/local-control-snapshot-20261018-r1/bc-probe/`，小证据同步到本地
`outputs/wam-rl-policy/scheduled-wam-bc-20261018-r1/`。运行前固定提交。

Limitations: 仅一个对象、两个raw episodes，成功终点筛选缓存不等于稳定保持数据。
teacher历史没有覆盖原生driver的前6步零动作初始化，也没有学生偏离分布。若后续
执行失败，须分别检查启动与保持；离线正向不能直接授权长预算PPO。

## 结果

代码28c11d6a，远端RTX4090完成128行真实WAM条件重算与1000步BC，39.74秒，
PyTorch峰值459.76MiB。TRAIN L1=.003093，内部dev L1=.291687，常量基线=.222350，
dev裁剪比例9.505%。未通过dev优于常量的预设门槛，不扩大该模型PPO。
状态/τ归一化后dev绝对z最大6.81/5.88，p95为2.31/1.93，均无|z|>10；
没有发现明显归一化数值爆炸，但这不能排除分布覆盖不足或任务信息不足。
本次小包不能否定BC→PPO路线，也不能把失败归为WAM。

后续只做原生12tick工程smoke（seed0、最多180秒、同一Camera/scale.06），验证
前6步零命令收集真实历史后能执行WAM条件BC命令，检查action隔离、τ记录和末次状态。
不更新参数，不作成功率比较，不因这个短smoke产生新的学习或抓持结论。

原生smoke完成：12tick，前6步零动作、后6步每步重新调用调度WAM；9.36秒，
action/prefix完全一致，末次观测与τ已存档。6次WAM前向为.531/.332/.214/.164/.333/.073秒，
与主训练共享GPU且样本极少，不视为稳态吞吐。无抬升/达标，但12步不用于策略质量判断。
第一次启动因未把conda bin加入PATH而找不到ninja，修正PATH后成功；不涉及模型失败。
该run旧manifest的note错误沿用了“GT-tau BC”文案，实际policy路径是本卡刚训练的
`bc-probe/model.pt`（conditioning=scheduled-wam-v4）；原始记录保留，后续runner已
改为从checkpoint读取conditioning，并将未运行oracle的replay指标设为null。
