---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-native-policy
experiment_id: P-20261015-shadow-native-policy
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: c6bacb7d
claim_id: C1
hypothesis_family: HF-shadow-native-policy
probe_index_in_family: 1
seed_pool: probe
seeds: [210, 211, 212, 213]
decision_changed_if_positive: expand native Shadow BC and matched data-scale success comparison
decision_changed_if_negative: diagnose native action wiring or fitting before scaling dataset
status: RUNNING
run_id: shadow-native-pilot-20261015-r1
---

# 在ShadowHand自身环境中检验WAM动作微调

Class: Decision；服务独立执行底座，用户明确要暂不做Inspire迁移，而是Shadow微调后直接测自身抓取。
主假设：既有采集控制数据能监督WAM-v2 shared+Shadow actor并闭环执行。
最便宜先复用原20对象raw、7TRAIN物体41条/6232窗、2DEV物体10条/1520窗，验证4-step与1000步pilot；
不将这个小pilot冒充全2066物体/33056轨迹的数据量实验。shadow成功只能说明该域可行；
数据不足解释需后续Shadow小数据/全量matched比较，不能单独归因Inspire失败。

协议与输入/标签/时钟/归一化/指标见[Task README](../../../README.md)。TRAIN-only归一化，
同80k WAM父模型、新Shadow actor、hand+scene forecast保留、原生8×24控制、无GTτ warmup。
状态是完整当前300D，goal是决策已知任务目标平移（保留当前朝向），不把未来物体观测用作goal。
数据候选按terminal距离<=5cm选，源内train/dev按mesh独立；不以dev模型误差筛数据。
固定2TRAIN+2DEV native面板、seed213、16env，每角色一个episode，teacher/student初始状态核对。
官方sticky success与final/hold代理分别报，不混为同一指标。工程4-step结果不评价方法。

## Decision Note / 用户资源例外

用户已明确“先写代码…暂时占用一张空卡跑小规模测试”，目录指定src/task/shadow-finetune。
因此原6卡训练继续，临时GPU6进行本Task小测试、GPU7空闲；该轮允许突破常规留两卡规则。
不暂停旧训练、不占远端GPU、不改原始数据。新GPU测试只限pipeline1h、14GB、train1000步，
准备限30min/16GB，剩余磁盘<20GiB、非有限、源码变动、子进程失败/超时则停止own进程组。
原teacher checkpoint仅作label来源与baseline，从不初始化学生actor。
首个science pilot成功/失败均先核对teacher校准、初始配对、原始command重建和拟合，再决定扩大。

## 验证与局限

3项合同测试覆盖60/30Hz对应、startup重复、未来扰动不改变输入、语义点顺序、动作梯度达WAM。
GPU首轮尚待启动。具体commit/命令/status保存在各run manifest/process；源码准备阶段git_commit
记录前一调查提交，启动后补实际代码身份。不形成正式成功率或data-size归因结论。
