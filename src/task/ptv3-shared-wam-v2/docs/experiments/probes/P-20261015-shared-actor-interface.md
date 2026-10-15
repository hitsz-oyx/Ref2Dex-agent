---
schema: ref2dex.probe.v2
probe_id: P-20261015-shared-actor-interface
experiment_id: P-20261015-shared-actor-interface
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 4a712b57
claim_id: C1
hypothesis_family: HF-shared-action-interface
probe_index_in_family: 1
seed_pool: probe
seeds: [190, 192]
decision_changed_if_positive: prioritize shared-token action interface for native grasp qualification
decision_changed_if_negative: inspect fitting and rollout quality before additional interface scaling
status: RUNNING
run_id: shared-actor-interface-20261015-r1
---

# Shared表示直接预测q_PD：匹配三臂

Class: Decision；当前仅离线Probe，结论待运行。服务独立执行底座；用户参考
[ref1](../../user/ref1.md)要求测试shared压缩表示能否改善动作。检验同一预训练WAM的
显式预测τ接口与直接shared token接口是否产生可决策的动作误差差异。
最便宜方法是共用现有native660、不新采集，固定父权重与三组训练预算。

## 固定协议

先提交数据变更27bd9913。之后新三臂全部使用补采native660与严格位置准入：
actual object相对reference位置>0.1m或非有限，将坏帧及前50帧隔离；任何H4至future24
support相交的窗口均剔除。workspace后TRAIN93680/dev10376，startup7205/971，
有窗序列546/589与68/71。原数据/manifest与人工标签保持不变。
审计`outputs/ptv3-shared-wam-v2/rollout-admission-current-20261015-r1/report.json`。
相对原122299窗减少18243；新dev改变，旧结果不作为单因素对照。

共同父权重remote-parent-20261015-r1/step80000.pt与stats_step80000.json，
预训练normalizer保持不变，actor state/tau/action normalizer仅从严筛TRAIN拟合。
无GTτ warmup，不从旧robot微调12k初始化，不读取teacher actor参数。
A=state+pred_tau，B=state+shared，C=state+shared+pred_tau。
shared为一次encode获得的768D融合token及原valid mask；保留scene/hand forecast损失。
动作标签仍是同tick q_PD编码8×15 geometry-continuous-lift；输入不额外加入teacher
1442D未来参考。WAM现有goal输入沿用父模型合同，不能宣称全输入只有当前观测。

actor三臂模块/名义参数/随机初始化相同：共享768→128→128memory投影，state87→128→128，
τ标准化后右补0至768，独立24步时间嵌入与state/τ/shared类型嵌入，8 queries，2层decoder，
128width/4heads/FF512/dropout0，15D零初始化输出。shared保留mask，不flatten。
零填充使τ的有效投影秩和shared不同，不能声称消除所有容量因素。shared-only未用的
τ时间参数由DDP find_unused_parameters处理；不影响三臂名义参数数目一致。

统一anchor_weight=0，防止GTτ特权监督造成不匹配；A是新的匹配τ baseline，不是旧R。
原τ路线action loss本已经τ反传WAM；本实验检验接口收益，并非首次动作监督WAM。
head零初始化第一更新只动readout，之后动作梯度才经接口到WAM；单测以非零readout核验。

单一seed190，训练抽样与初始化共用，panel192；三臂每卡256/双卡global512、12000更新，
lr geometry5e-6/shared1e-5/actor5e-5，AdamW/AMPbf16/TF32，梯度裁剪1，forecast/action权重1。
相同序列均匀+startup/later均衡TRAIN抽样，固定新dev256索引，500步评价。
记录全部hand/scene/action/anchor与总损失；物理first-control XYZ mean/p95、旋转测地误差、
finger与Euler lift误差、控制饱和比例、hand/scene all24误差。离线无闭环/真实success结果。
保留2k/4k/6k/8k/10k/12k完整权重、最近点、best total、best XYZ p95、best forecast。
选择规则预先固定，报告同一步12k与各自selected，不能只报最优臂有利的阶段。

判别：同一步12k上B/C的first-control XYZ mean与p95相对A均降低>=20%，且forecast合计
均值无>20%退化，标记局部PROMISING，下一步native桥接+匹配抓取验证；三臂拟合仍差
或差异小则UNCLEAR，先查TRAIN/dev泛化或动作拟合，不直接证明τ存在不可消除信息瓶颈。
离线结果不升级全局claim；早期4-step仅工程smoke，不用于科研判断。

## Decision Note / 资源与停止

用户刚更新CAMPAIGN：本地始终留2卡空闲，远端可全占。故本地GPU0/1、2/3、4/5分别
三臂同时执行（6卡，6/7空闲）；远端既有四卡不受影响。不再使用已废弃的全局六卡总计。
先每臂双卡4-step batch128工程检查已全部通过（峰值5.5GiB），统一增至batch256再次检查后固定代码detached工作树长训。
三臂各<=80000s、14GB，总42GB，CPU loader每rank2workers，20GiB磁盘余量。
非有限、外来GPU进程、无更新1800s、超时/预算或源码漂移终止自己的进程组。
不覆盖历史checkpoint，不重启Viser8768或改动人工审阅。首轮监测真实吞吐/显存后给ETA。

## 身份、运行与局限

代码commit见运行manifest（本文git_commit记录模型实现，smoke启动器随后补充）；本地main含其他任务未提交
变更，运行固定快照仅包含本实验提交及既有已提交依赖。run_id使用共同前缀加接口后缀。
新head暂不被legacy τ物理执行入口接收，明确fail closed，正向离线信号后再实现桥接。
本Probe不排除容量/优化/数据覆盖，不动态使用人审标签，不增训已有旧baseline。

## 工程预检

实现4a712b57，smoke启动器21dd46c8；28项相关测试通过，独立review修复shared-only
DDP unused时间参数后通过。三臂双卡batch128的4-step全部COMPLETED/exit0，
输出shared-actor-smoke-{pred_tau,shared,shared_tau}-20261015-r1，完整模型/optimizer已保存。
初步每卡峰值allocated5.5GiB，保留GPU6/7空闲；为增加吞吐，正式匹配batch改为256/rank。
工程步骤不是拟合收益结论。
