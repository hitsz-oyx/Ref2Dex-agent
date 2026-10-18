---
schema: ref2dex.probe.v2
probe_id: P-20261018-scheduled-batch-ppo
experiment_id: P-20261018-scheduled-batch-ppo
date: 2026-10-18
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 3
seed_pool: debug
seeds: [0]
decision_changed_if_positive: expand the scheduled BC warmstart PPO to a training object panel
decision_changed_if_negative: retain BC and repair continuous PPO before scaling
status: COMPLETED
run_id: scheduled-batch-ppo-20261018-r1
---

# Scheduled BC 的批量 PPO 试跑

Class: Decision。区分持续 on-policy 更新是否可运行且保留基本控制，与单次工程更新
不能暴露的累计退化。先在一个物体的多环境运行，是扩展训练对象池前最便宜的判别。
不把此结果解释为跨对象成功率或 WAM 条件的因果增益。

初始化：全量 scheduled BC 的 fit-fast/best.pt (step46000)，冻结 WAM step12000。
每拍每环境独立 scheduler/H4 历史，就绪环境合并一次 WAM forward。
565D current+world tau+valid，24D unclipped Gaussian 用于 PPO likelihood；
原生动作 clamp 后另取副本交给模拟器，native reward 不改。

首轮 GPU6，16环境，180拍/rollout，4次 PPO update，各2epoch/4minibatch，
actor LR1e-5、critic LR1e-4、joint KL上限.01；前后各一次确定性均值策略评估。
每条 rollout 在 native terminal 前显式截断，以实测末态 bootstrap，不把截断当失败终止。
随后显式重置全部环境/历史。若提前出现 native terminal 则失败退出，不跨 episode 拼历史。
评估复用 seed0，训练 reset种子为 seed0+update；GPU physics 不承诺完全相同轨迹。
这会限制长时保持结论，后续需要真正跨 chunk 持续 episode 和异步 reset。

预算1800秒、磁盘余量20GiB、最多2个PPO端点；STOP文件/非有限/概率回放漂移/
无接受的actor更新停止。父BC永不覆盖。用户已允许共享GPU；GPU1/7不分配本任务。
环境 graspenv，Python3.8/Torch2.4.1+cu121/IsaacGym，运行版本写manifest。
输出 `outputs/wam-rl-policy-v2/scheduled-batch-ppo-20261018-r1/`。
入口 `tools/run/closed_loop_scheduled_shadow.py --envs 16 --ppo-updates 4 --ticks 180`。

批量历史的部分重置、合并推理、60Hz插值以及原控制合同/目标指标共4项测试通过。
运行结果待记录。

启动记录：代码03a696eb，GPU6/16环境，物体 bottle-109d55a137c042f5760315ac3bf2c13e
确认属于TRAIN池。日志 `tmp/scheduled-batch-ppo-20261018-r1.log`，
运行初期GPU约55–59%/3.6GiB，磁盘剩余149GiB。

r1在第二次更新前被严格likelihood回放检查拦下(1.4687e-4)。固定checkpoint和
归档565D输入的3秒复现：合并2880条重放误差1.1444e-4，按采集16条重放误差0。
修正为逐采集tick批大小回放，同时独立核验存储Normal分布；阈值仍1e-4。
21项测试通过；r1保留首个PPO端点，r2从同一BC重新开始完整试跑。

## r2完成结果与决策

实际代码 `bb823d64`，输出 `outputs/wam-rl-policy-v2/scheduled-batch-ppo-20261018-r2/`。
全程245秒，训练11520 transitions，另有前后各2880条评估交互。
4次PPO更新分别接受3/6/6/7个actor步，共22步；critic各8步。
回放误差全部0；KL分别.00907/.00980/.00984/.00921，均低于.01。
actor LR首步校准至2.5e-6；critic解释方差.140→.685。
GPU观测55–67%、总显存约3.7GiB，PyTorch峰值584MiB；完成后GPU6回到共享任务347MiB。

同seed前后初始300维观测逐值一致；BC→PPO均值评估：
- 终态距目标<5cm：2/16→4/16。
- 平均终态距离：23.88→22.72cm。
- 平均native reward：.22369→.28443。
- 最长连续达标帧数的最大值：78→95 (60Hz)。

r1/r2的BC初始观测也逐值一致，但BC终态达标数4/16与2/16不同，
证明当前物理轨迹不可视为严格配对的确定性复现。上述前后变化不构成PPO提升证据。
大部分环境仍未完成抓取；最大保持长度不能代表全部环境的稳定成功率。

端点：`ppo-update-1.pt`、`ppo-update-4.pt`，包含actor/critic/optimizer和条件合同；
原BC不覆盖。`baseline.npz`、`final-evaluation.npz` 保存逐拍距离与初态，
`result-summary.json` 汇总结果。

决策：工程blocker已消除，可以进入TRAIN多物体panel和更长预算；本轮不因
单次2/16→4/16就发布策略提升结论。下一步需补PPO端点加载/续训入口、多物体
几何与环境身份绑定、跨chunk episode/reset处理，再扩大训练；评估加入重复BC对照。
未实施C条件或phase条件，也未把空间goal重新命名为已验证的0.8秒可达目标。
