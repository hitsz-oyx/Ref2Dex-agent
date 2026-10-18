# UniDexGrasp：调度 goal → 冻结 WAM-v4 → BC 初始化 PPO

用户于2026-10-18指定以 `docs/user/架构.md` 为架构，先用 UniDexGrasp 数据与环境，
执行器使用自训 BC 初始化 PPO。此限定适用于后续控制数据和评估；已经在跑的九源
WAM预训练不改配方。目标仍是独立策略和表示增益，UniDex结果不能自动代表DExplore
参考轨迹任务已经完成。

## 已固定的接口

- 目标：UniDex任务提供的物体目标位置。初态到任务目标构造已知指令路径，初态旋转
  在本轮位置任务中保持；不能用执行后录到的未来物体轨迹伪装成在线任务目标。
- 调度：复用更新后的 `OrderedGoalScheduler`，含手物历史proxy、进度和恢复约束。
  每环境独立状态，每次reset重新绑定该次任务路径。几何完成不等同抓稳。
- WAM：固定修复版v4 step12000（开始下载时DEV best），原始完整checkpoint保留，
  移植几何路径仅发生在独立推理副本。当前实测7帧60Hz按0/2/4/6取H4，预测30Hz
  24步；前8步60Hz世界系手位移送执行器。原32个flow query身份保持不变。
- 执行器：初始兼容现有 `p2_tau` BC，139D局部状态+264D手轨迹→24D Shadow动作。
  这只是首轮兼容合同，partial observation不是最终执行器必须限制；不要把它的优势
  当成完整状态下的表示增益。物体轨迹和C由运行时保留，首轮actor不消费C，后续
  `state+tau` / `state+tau+C` 使用相同训练预算做独立对照。
- BC→PPO：保留BC权重和输入归一化，actor整体可训练，单独初始化critic与探索std。
  BC不是永久动作基座，不加载官方actor权重作为最终策略初始化。官方策略可以作为
  TRAIN动作标签提供者，必须标记来源。PPO储存裁剪前动作及其logprob，执行端裁剪
  并复制动作，避免上游原地旋转污染历史。

## 已实现与未实现的边界

`wam_rl_policy.scheduled_executor`提供单环境调度→v4→学生观测，以及BC初始化的
Gaussian actor。合同测试检查BC输出完全一致、PPO概率重放与更新、归一化冻结、
H4抽样、调度goal传递与reset。现有 `recipe_ppo.update_minibatches` 支持其
`independent-gaussian` 动作类型；这不是完整原生PPO runner已经接通的证据。

原 `official_unidex_ppo.py` 的2607D summary路径不作为此路线的τ/C执行实现。
原生确定性runner现已有`--wam-checkpoint`模式，12tick工程smoke通过；启动6步使用
零命令获得真实历史。实际WAM条件数据重算和小包BC也已完成，内部开发未过门，见
[实验卡](../experiments/probes/P-20261018-scheduled-wam-bc.md)。尚需PPO原生rollout、
done/time-limit bootstrap及训练/部署启动分布一致性。观测全部来自同一世界系，
原生动作保持canonical；不重复当前帧伪造已观测运动历史。

## 下一次有界 Decision Probe

问题：在相同UniDex任务和实测状态下，计划条件执行器是否能通过自训BC获得可用起点，
并在PPO后改善连续保持？最便宜的次序是先真实单环境接线，再小批量训练。

1. 固定TRAIN对象和初态分组、DEV对象和初态；历史调试421/453/471/482不能冒充
   未触碰TEST。首先复用已存在UniDex资产，检查许可、动作标签和几何身份，不新下载。
2. 在TRAIN实测状态上运行同一调度器和冻结WAM，配对应的真实动作标签，训练BC。
   GT-τ仅作独立上界；不能只在GT条件训练然后把WAM替换的退化归为执行器失败。
3. 原生单环境工程smoke检查真实输入、动作、reset、终末观测和耗时；记录WAM延迟，
   再决定重规划频率。若降频，必须显式移位/重锚计划，不能重用旧相对τ冒充新预测。
4. 固定BC checkpoint，比较确定性BC与相同起点短预算PPO，记录每次实际动作、
   连续in-goal帧、抬升、末次动作后的距离和掉落；累计ever-success单列。
5. 通过后才扩大对象/环境数，并做state-only、tau和tau+C匹配对照。接口或保持失败
   时先定位，不直接加长同配方训练。C是并行交互预测，不能当动作反事实critic。

每次训练前建立Task-local实验卡，固定代码、输入、run_id、GPU/时长/产物预算与环境。
当前这次工作为下载和工程整合，不宣称新的物理成功或PPO收益。
