---
schema: ref2dex.probe.v2
probe_id: P-20261015-shadow-native-policy
experiment_id: P-20261015-shadow-native-policy
date: 2026-10-15
task: shadow-finetune
branch: main
git_commit: 5cf32aa5
claim_id: C1
hypothesis_family: HF-shadow-native-policy
probe_index_in_family: 1
seed_pool: probe
seeds: [210, 211, 212, 213]
decision_changed_if_positive: expand native Shadow BC and matched data-scale success comparison
decision_changed_if_negative: diagnose native action wiring or fitting before scaling dataset
status: UNCLEAR
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

4项合同测试覆盖60/30Hz对应、startup重复、未来扰动不改变输入、语义点顺序、动作梯度达WAM，
以及抬升与保持必须同时满足。实际运行代码5cf32aa5；具体命令/status保存在run manifest/process。
不形成正式成功率或data-size归因结论。

## 已执行的工程核验

代码5cf32aa5，supervisor2028217，GPU6，实际四步完整模型smoke已完成，峰值allocated1799MiB。
paired native16env smoke完成，初始q/object/hand及独立扩展dq/root/observation300/goal/reset均差0；
学生分支controller-audit明确student=true，无invalid fallback。teacher sticky5/16、terminal4/16、
位置held60+lift3/16；四步学生均0，不作拟合能力负证据。TRAIN/DEV分别计数已由raw后处理输出，
其源脚本在项目tmp/shadow_native_group_metrics_20261015.py。原student result.json继承collector
policy文本不准确，实际角色以controller-audit和comparison.json为准；原始结果保留，后续版本修正元数据。
独立审查153个状态复算官方actor与label max2.861e-6，上一拍action观测完全对应，无teacher泄漏。
细节tmp/shadow-semantics-review-20261015-r1/review.md。本轮3项合同测试及verify通过。

## Pilot完成与下一步边界

1000步pilot及终点评估均完成，训练568.3秒、峰值allocated2999MiB，GPU6已释放。
250/500/750/1000固定DEV action归一化L1分别0.23122/0.21082/0.22324/0.21421；
按该预定指标选择step500。对应首拍原生命令MAE分别0.12697/0.09541/0.09113/0.08197。
小batch训练采样利用率约14%，大规模运行前应改善CPU输入吞吐。

16env面板teacher官方success5/16，学生1/16；学生唯一成功在TRAIN camera，
TRAIN1/8、DEV0/8，teacher分别3/8、2/8。同步满足连续60帧距目标<=5cm且抬升>=10cm的
位置保持代理teacher3/16、学生1/16。所有初始状态字段误差0，学生无invalid输入。
原始raw/comparison保留；新CPU复算在
`outputs/shadow-finetune/shadow-native-pilot-20261015-r1/pilot-native/simultaneous-hold-split-audit.json`，
按对象和split分别报告，避免把一次TRAIN成功当作DEV泛化。后续evaluate修正继承的policy文本，
完整核对初始字段，并要求抬升和保持发生在同一时段；这不改变本次raw轨迹。

判断UNCLEAR：原生监督与闭环链路可运行，但小样本尚无DEV成功，不能检验数据量解释。
CPU已从全部83个已有raw replay准备完整候选cache，耗时183.9秒；未启动新的全量GPU训练。
`outputs/shadow-finetune/policy-data-full-20261015-r1/counts-and-split-audit.json`记录：
TRAIN1567物体/10118轨迹/1536897窗，DEV388物体/2338轨迹/354761窗，物体交集0。
这是终点5cm及运动筛选后的12456候选轨迹，不是全33056轨迹均参与；34条无完整安全窗被拒绝。
下一步需要同域小/大数据matched Probe，保持优化预算、heldout面板和reset一致；本轮临时单卡授权
只用于已完成的小测试，不自动扩展长期占卡。未来证据：更充分拟合、多seed和稳定接触认证。

完整cache进入下一轮前执行工程Blocker审计：逐条验证raw manifest的物体/scale/split和CAD归属、
60Hz状态/上一拍命令对应、H4/K24与8动作完整支持、已声明的速度隔离/尾段规则和物体划分无交集。
使用纯NumPy文件审计，CPU≤10min、新报告≤200MB，不调用模型、不占GPU、不更改cache。
最便宜先在51条pilot cache核验入口，然后遍历12456条full cache；发现任何差异则先修适配，
不把后续拟合失败归因数据量。这只验证数据合同，不代替teacher全量推理或物理抓持认证。
