# WAM robot BC→RL

实现冻结WAM/BC reference上的小型在线控制器。路线输入见
[用户参考](docs/user/ref/ref1.md)；当前先做Shadow Stage-0 residual PPO，
之后才根据结果决定RL-token readout与独立动作支路。

`src/wam_rl_policy/ppo.py`定义327D观测、24D canonical残差与PPO；
`shadow_loop.py`连接既有原生Shadow BC，`tools/run/shadow_ppo.py`复用固定collector
的环境初始化和物理循环。代码不修改第三方物理实现。

当前Stage-0已完成20次真实PPO更新和三对reset评估，保持11/48→15/48但方向不一致，
结论UNCLEAR；冻结与原生执行审计通过。详见[当前状态](docs/STATE.md)。

使用`/home2/wyy/miniconda3/envs/graspenv/bin/python`，入口先加载IsaacGym再导入Torch。
`--mode bc/zero/train/evaluate`分别进行BC、零残差、PPO和确定性PPO评估。
TRAIN-only在线探索，DEV只用于固定终点评估；固定原官方reward。
研究协议与输出路径见[首轮Probe](docs/experiments/probes/P-20261016-shadow-bc-ppo.md)。
