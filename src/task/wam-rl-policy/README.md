# WAM robot BC→RL

2026-10-16新增[任务读出与目标语义Probe](docs/experiments/probes/P-20261016-wam-task-readout.md)：
`task_readout.py`支持状态/目标查询、真实短期物理辅助监督与冻结预测输入，不依赖BC动作
监督。新读出未超过原mean，进一步定位到预训练短期终点条件与整任务goal的差异；
下一步优先短期目标桥接，BC不再作为研究主线前置关卡。具体结果见[当前状态](docs/STATE.md)。

实现冻结WAM/BC reference上的小型在线控制器。路线输入见
[用户参考](docs/user/ref/ref1.md)；当前先做Shadow Stage-0 residual PPO，
之后才根据结果决定RL-token readout与独立动作支路。

`src/wam_rl_policy/ppo.py`定义默认327D观测及可选539D冻结表征观测、24D canonical残差与PPO；
`shadow_loop.py`连接既有原生Shadow BC，`tools/run/shadow_ppo.py`复用固定collector
的环境初始化和物理循环。代码不修改第三方物理实现。

当前已按用户授权启动128个TRAIN物体的四卡同步PPO，200轮目标、50/100/200轮固定DEV
读出；另完整训练fresh单拍BC MLP并完成388DEV/1552起点native基线，held396/1552。
同起点WAM9k held511/1552；远端MLP PPO200轮和探索噪声对照尚无明显改善，
正在检查等容量raw归一化输入与BC隐藏特征；此前两物体Stage-0仍UNCLEAR。详见[当前状态](docs/STATE.md)。

使用`/home2/wyy/miniconda3/envs/graspenv/bin/python`，入口先加载IsaacGym再导入Torch。
`--mode bc/zero/train/evaluate`分别进行BC、零残差、PPO和确定性PPO评估。
TRAIN-only在线探索，DEV只用于固定终点评估；固定原官方reward。
研究协议与输出路径见[首轮Probe](docs/experiments/probes/P-20261016-shadow-bc-ppo.md)。

规模入口为`tools/run/launch_scale_ppo.py`；`prepare_scale_panels.py`固定对象和资产，
`evaluate_scale.py`执行完整DEV/阶段面板，`run_scale_queue.py`监督有界读出队列。
`train_current_bc.py`整理同拍标签并训练MLP；`--bc-kind mlp`连接相同native PPO。
参见[规模Probe](docs/experiments/probes/P-20261016-shadow-scale-continuous.md)。
