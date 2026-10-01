# 从局部防丢失转向可学习的动作选择

2026-10-01，HF11，Decision；HF10 2/2关闭，不修改其失败判定。

问题：Cm是否能将已观察到的短期保留信息转成可学习的实际决策，而不再
只通过微小均值辅助损失影响actor？证据是新五推荐器3821窗口：已抬升状态
release标签vsbase −5.470pp，描述性frame90[−8.167,−2.774]，vsstate-only/
shuffled也降低；但保留高度的四控制收益门失败，vs固定cup的risk差仍不确定。
这仅支持局部恢复机制继续探索，不能当稳定抓取或HF10通过。

选择：冻结物理Cm及六个自训练专家，在相同2步候选+8步base+6步cooldown
合同上，训练一个观察驱动的六类片段选择策略。Cm直接提供22维物理未来
特征及明确动作先验；PPO优化六类选择概率，实际执行选中的片段。评估训练
所得选择策略仍调用Cm，使作用链在推理时保留。训练对象是新的选择策略，
不是重新包装冻结控制器，也不更新旧actor/V。四控制机制数据保持只评价。

Cm-on/off：完全相同新策略网络、初始权重、六专家、模拟器配置、优化器、
训练交互预算和原生reward。on使用冻结Cm的候选未来和推荐动作先验；off
计算同一Cm以匹配推理成本，但屏蔽物理特征，先验推荐base。两者先验均为
推荐动作.90、其余各.02，保证初始熵相同且保留探索。政策输出为
softmax(可学习logits+log先验)，实际categorical option ID的概率进入PPO；
不得先抽Gaussian动作再覆盖，并继续使用原Gaussian logprob。

输入：10步物理历史、当前六组12独立控制通道、base/motion/phase上下文；
on额外使用候选10步高度/接触及联合接触、release预测，off同形状置零。
NN归一化固定；新策略历史norm裁剪到±10防极少velocity尾部，Cm本身不改。
策略为GRU32+共享候选评分头，独立MC value head仅帮助PPO估计优势；
完整episode的折扣原生reward作MC目标，无历史V或Cm制造的reward标签。
未完成episode不得用于训练或稳定结果。PPOclip.2、gamma.99、lr3e−4、
20epochs/rollout、minibatch128、entropy.01、value.5、gradclip1。

最便宜步骤先做工程smoke：验证初始先验、实际选择logprob/ratio、有限更新、
冻结专家/Cm及2+8执行合同。通过才运行单训练seed381，4个完整96env rollout/
arm；每rollout只在结束后更新，下一rollout由学到的策略实际执行。记录所有
optimizer更新、动作概率变化和原生reward，不把loss当utility。

独立评价seed391–394，各96env完整首episode。先固定稳定指标：高于初始
rest3cm并有原hand/object力代理连续45ticks（1.5秒），达标后继续到首episode
结束检查掉落；stable-success=曾保持且随后无跌至2cm/连续6tick失去接触。
同时报告原五步保持、lift acquisition、post-lift release和实际选择覆盖。
原始未训练先验控制器在相同seed至少一组评价，以区分学习和冻结控制器作用。
动作支持、合同、有限梯度是工程门；单训练seed的stable收益只标Probe。

研究screen：训练策略参数与实际决策均改变；on/off各预算完整且matched；
Cm-on稳定成功率高于off，已抬升后的释放率不增加超过2pp，未训练先验对照
不能解释全部学习收益。结果及区间完整报告，不按中途reward提前选checkpoint。
无正信号则不追加rollout/改reward/换seed追门，HF11 slot1关闭并回到物理输入
及候选片段规划层面。若PROMISING，再独立预注册多训练seed Validation，
不提前形成C3结论。

预算：HF11研究Probe1/1，工程smoke不占科学slot但成本计入；总≤60分钟/8GiB，
最多2张空闲GPU，已有机器最多4GPU边界不变。代码/输入先冻结再启动，
所有输出唯一且保留旧证据。失败停止本地调参。MISSION/claim不变，无新授权边界。
