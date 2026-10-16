# 当前状态

2026-10-16：Shadow BC→RL Stage-0完成，UNCLEAR。代码在本Task，路线来自用户ref1；
冻结Shadow9k BC/WAM，只训练新327D actor/critic。20×199×32=127360 env steps，
约19.2分钟；80个actor epoch均接受，jointKL<=.01762，latent logprob重放最大5.73e-6。
全部597个冻结state_dict字段逐值未变、无gradient；canonical/yaw/force/torque误差0，
手指目标最大1.2e-7。

三个同初态16env reset面板216/233/234，BC/PPO保持分别5/8、4/3、2/4；合计11/48→15/48，
DEV1/24→3/24，official sticky均26/48。方向不一致；同一BC本身还有9/16与6/16的数值
重复波动，不能据合计+4宣布有效。保持为60帧goal<=5cm且lift>=10cm位置代理，非接触认证。
一个训练seed、仅两TRAIN物体，不作正式结论。

完全同初始输入/冻结权重/fixed RNG，GPU reference重复推理仍有约6e-6–1.12e-5 command差；
zero same-packet精确等于BC，但独立闭环不能要求逐值相同。原完整轨迹门FAIL保留，
未定位具体算子，也未排除额外PhysX波动。独立审查已复算全部raw分split指标与初态。

下一步先诊断有效探索和reward/保持关系，检查饱和边界是否吞掉残差，再决定是否扩大
PPO预算；不要因合计增益直接放开WAM。learned RL-token、aligned/shuffled/none和独立
动作路径仍是后续路线，当前尚未实现/验证其效用。

产物`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/`；模型`train/latest.pt`，
评估`evaluation-pairs.json`，汇总`summary/results.json/csv`与`comparison.png`。
checkpoint保存actor/critic/optimizer/RNG；CLI仅evaluate加载actor，未实现train-resume。
全部有界作业退出，未停止外部GPU0任务。协议与证据见
[P-20261016-shadow-bc-ppo](experiments/probes/P-20261016-shadow-bc-ppo.md)。
