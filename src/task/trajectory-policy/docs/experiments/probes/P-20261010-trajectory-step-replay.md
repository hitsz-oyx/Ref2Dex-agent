---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-step-replay
experiment_id: P-20261010-trajectory-step-replay
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-optimization
probe_index_in_family: 1
seed_pool: probe
seeds: [293]
decision_changed_if_positive: test bounded adaptive actor step scaling with unchanged task contracts
decision_changed_if_negative: inspect action dimension and gradient allocation before more training
status: UNCLEAR
run_id: trajectory-step-replay-20261010-r1
---

# Does persistent early LR backtracking constrain later useful updates?

## Motivation / Decision Note

服务于Mission A的可用自训练baseline，随后仍需Mission B matched Cm训练收益。
lambda1确实改善startup优势，但mean0/4、冻结同噪声warm/final同2/28，且同pair
成功；mean变化在1mm探索尺度下很小。不是通过追加epochs继续碰运气，而是区分
当前优化是否存在可改变决策的步长瓶颈。分类Decision，最便宜方法是保存batch回放，
无需新的环境交互、teacher、BC或WM。

三个可检验的假设按优先级：
1. u0回退后的6.25e-8 LR永久保留，限制后续可接受更新；恢复1e-6再按同KL guard
   回退，能在多数真实before-state获得更大mean movement和更好同batch surrogate。
2. 全288维joint KL使动作/后16步占用更新预算；报告first8/suffix及XYZ/rotation/
   fingers KL分解，若恢复LR也无更大有效更新，转向维度/梯度分配而不盲目改LR。
3. 共享网络的更新不能有效保留positive startup样本；固定全部12条>=45held探索行，
   比较实际startup logprob变化与优势符号。单batch改善不等于真实抓持收益。
另报告FP32 log_std实际变化坐标数，防止把声明trainable当成已经改变探索。

Root选择一次有界机制Probe。旧训练packet没有Adam moments，必须从u0空Adam
重建全部原updates，并**逐轮bitwise核对actor/value checkpoint**后才解释fork；
不能用fresh late optimizer替代历史状态。预计~10--30s GPU，无新策略产物。
正向才设计一次有界真实任务步长Probe；否则先查动作/梯度分配，不重复lambda/
BC权重/48D搜索，不改Mission或启用WM。重建失败保留FAILED，先定位差异，
不把fork作为证据。无需要外部授权的新边界。

## Fixed protocol

输入：`outputs/trajectory-policy/trajectory-ppo-long-credit-20261010-r1/`
完整high/low/monitor、u0:23与final；物理/输入/原ppo.py/source hash不变。
H来自`outputs/trajectory-policy/trajectory-stochastic-execution-20261010-r1/`
已审计plans，沿用当前source hash。旧mean执行器源码在增加paired模式后已变化，
不放宽旧hash guard或把旧源码视作当前相同代码；诊断固定使用新wave的测量H。
标准Torch2.4.1+cu121、GPU4、highest FP32、2threads。
actor Adam初始1e-6、critic3e-4，原四epoch PPO/.2clip/.5actor gradnorm/
joint KL<=.02、四次.25回退，lambda1、真实duration8/6和原bootstrap。
原log_std clamp也逐轮重放。每个before及after actor/value全部state_dict exact，
actual accepted epochs/LR和stored优势exact；失败立即停止，不放宽要求。

对全部24个真实before-state各fork一次：相同actor/value、相同恢复的Adam states，
只将actor LR恢复1e-6，随后调用原ppo.update，保持其他学习合同。fork只进行这一
update，不进入下一批数据或环境；历史臂仍原顺序重建并独立继续，禁止moments alias。
没有跨update的反事实训练或将fork policy称为on-policy学习。

报告每臂accepted epochs/LR/KL、first8/suffix和物理坐标KL分解、clip ratio、实际
post-update clipped surrogate。固定本轮同噪声执行wave全部warm tick0:8:40的168个
实测H，用同H比较before/after first8 XYZ rawmean变化，单位mm per-coordinate RMS；
不输入未来reference，不声称变化必能形成可执行τ或抓持。
从完整low held>=45连续行选择**全部12条**已知startup queries，不挑成功案例；
报告对应优势、实际样本logprob变化。保存所有24update记录与完整输入SHA。

探索screen：>=12/24 fork具有>=2倍precontact XYZmean movement、same KL<=.02
且同batch surrogate优于历史更新则机制PROMISING；否则UNCLEAR。
这是步长可用性线索，不是优化锁死、LR因果解释或正式RL收益。

## Resources / stop

一张空闲GPU4，新增3GPUmin/128MiB，单完整回放<=120s。GPU网络/Adam，CPU
文件/hash与结果统计，逐update输出进度。无新physics/seed/训练rollout/checkpoint。
foreign GPU冲突、非有限值、历史checkpoint/输入漂移、budget超限立即停止，保留
FAILED manifest和已经完成的记录，不自动重跑。原训练/checkpoint全部只读。

## Results

Not run yet; protocol frozen before diagnostic execution.

## Limitations / future evidence

单seed、一个imperfect warm start、lambda1固定采样数据；resetLR臂每次仅一update，
不代表持续采用该LR的on-policy数据分布。KL分解只是优化度量，不等价于任务因果
重要性；未来16步还会进入R的前缀conditioning。sameH mean位移与概率变化都不能
代替完整稳定held/drop评价。正向仍需真实任务Probe、matched native-action baseline，
最后Cm-on/off策略训练Validation。主目标仍未完成。
