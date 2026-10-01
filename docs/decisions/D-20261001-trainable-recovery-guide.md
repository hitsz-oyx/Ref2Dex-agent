# 可学习的Cm作用链与共同的保留目标

2026-10-01，HF12，独立Decision Probe1/1；HF11预算1/1与HF10预算2/2保持关闭。

问题：能否让已确认的局部保留信息帮助一个真正拥有执行权的可学习策略？
HF11诊断已区分“没学到分数”与“执行压住分数”：on最大相对分数1.283，
固定先验优势3.807；5065评价决策原始argmax全未变。on17/384vs13/384及
trained6/96vsprior3/96都不能称学习收益。另已确认原生reward是reference
body/object/IG/contact imitation乘积，没有显式保留目标。这两项是接法和
优化目标的结构缺口，不通过增加epochs或重评成功率修复。

行动：Cm冻结。23维未来特征=原22维物理未来+Cm推荐指示；off屏蔽22维未来
并用已知base指示。NN共享候选评分之外加入一个由PPO更新的guide_weight，
初始log5，得到推荐.50、其他各.10，明确一半利用/一半探索。没有固定外加
log先验或冻结权重；所有最终动作logits由可学习策略参数决定。NN仍在评价
时调用Cm；guide推荐作为输入信息而不是执行层永久偏置。两组同初始参数/
熵/架构/专家/新seed，PPO使用完整实际categorical概率。该联合路线不声称
单独分离guide ownership和reward变化；matched on/off只检验在共同新任务
目标下Cm信息的价值。

两组共同reward：当前hand/object力代理同时成立时
clip((object_z−rest_z)/.03,0,1)，否则0。每帧0..1，达到3cm后不奖励更高
运动幅度，持续接触支持才继续得到reward；丢失后自然失去后续保持reward。
原生imitation仍单独记录，但不进入本路线MC/PPO目标。固定gamma.99、reward
尺度.01使完整MC目标≤1；PPO其它配置保持.2clip/lr3e−4/20epochs/128batch/
.01entropy/.5value/gradclip1。只训练新选择策略及其MC head，六专家/Cm不改。

最便宜步骤先跑seed400的两臂完整96episode GPU工程smoke/2epochs，核对
guide确由梯度更新、没有固定先验、真实option概率/命令、bounded state reward
及完整MC/冻结合同。通过后单训练seed401、4完整96episode rollout/arm。
评价全新411–414、每seed各96完整episode，greedy实际执行；初始未训练
guide控制器另在411评价两臂。所有冷初始root/dof/rigid/force/observation/
motion/start/RNG及完整object/contact/action/reward/done trace都保存，
便于复算并确认观测、保持和后续drop边界；不改完备输入追表面配对。

先判学习机制：评价时至少5%的Cm-on决策产生相对初始guide推荐不同的12独立
控制命令；原始option ID同时报告。未达到则不把success变化归为学习。
任务screen预固定：stable45tick且随后无drop，on−off≥2pp，acquisition后
release率不增加超过2pp，并且trained-on在411优于frozen-guide-on。报告
acquisition与全episode release，后处理acquisition条件率只作任务描述，
不是共同处理前状态的风险因果效应。四eval seed描述性区间与所有分数完整
保留；单训练seed无正式claim。正信号才独立Validation，不扫lambda/reward/
epochs/seed。失败则关闭1/1，回到实际候选片段/接触几何层面，不局部调参。

成本：工程+Probe≤60分钟/8GiB，最多2空闲GPU（当前2/3声明，通常只用一张），
预计30分钟。MISSION claim/权限/deadline不变，无新的授权边界。实际全仿真
步数与有效首回合步数分别报告，所有旧数据/checkpoint/失败门保留。

Pre-science endpoint contract refinement: r1 engineering completed with both arms
updating the owned guide (210.164s/33.18MB), but source inspection found the
legacy H10-collection eligibility forced base during the last11episode ticks.
For HF12 only, actor decisions remain eligible until the actual native terminal
step. If fewer than a full H10 steps remain, zero Cm futures/base recommendation
(abstain in information, no forced execution). A cached2command may execute
a one-step prefix only at episode termination; record/audit executed_steps.
Revised r2 engineering must pass before science. Preserve r1 as completed
engineering with its original source identity and charge its whole cost; no
scientific outcomes or thresholds changed, scientific slot still0/1.
