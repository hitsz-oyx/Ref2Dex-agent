---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-shadow-parity
experiment_id: P-20261019-inspire-shadow-parity
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 11
seed_pool: probe
seeds: [246]
decision_changed_if_positive: continue the full-data Shadow-recipe Inspire PPO on the ordered object-path task
decision_changed_if_negative: repair the first failing robot adapter without changing PPO algorithm or shrinking BC coverage
status: RUNNING
run_id: inspire-shadow-parity-20261019-r1
---

用户修订：Inspire必须大部分遵从Shadow成功路线，不使用DExplore原ref输入PPO，
不再以小网络/单序列试跑替代正式BC路线。PPO任务使用完整已知物体SE3路径，经scheduler接WAM；
不是直接给actor人体手ref、参考姿态误差或未来实测手。原小BC结果保留为独立工程尝试。

Decision：全数据、同网络及同PPO算法在Inspire的适配是否能够稳定训练和执行？
最低成本的必要步骤是先完成无ref物理输入、全量BC及小批实际官方UniDex PPO更新/错开reset检查，
通过后才放大环境数；不能由单序列BC失败推断方法不成立。

保持项：共享NormalizedCommandMLP隐藏层1024/1024/512/512、ELU、零输出初始化；
TRAIN-only状态/动作std下限.05、仅current裁到[-5,5]、tau固定.05m缩放/valid尺度1；
先current BC再零扩展WAM条件BC、AdamW3e-4/weight_decay1e-5、标准化L1、batch2048、
两阶段各50000更新、半启动/半全TRAIN采样、同object-balanced DEV选择。
PPO直接调用Shadow的prepare_official和UniDexGrasp2官方引擎/GAE/clip/adaptive KL分布；
rollout8、5epoch、4minibatch、gamma.96、lambda.95、LR3e-4、KL.016、clip.2、初始noise.8。
不调用DExplore PPO/rl_games更新、不加载官方DExplore actor为策略基础。

机器人适配：18原生DOF、12独立命令、30Hz；H4[-3,-2,-1,0]，tau8×11×3世界系。
无ref current312=q18+dq18+physical248+object pose/velocity13+task endpoint xyz3+previous action12，
拼tau264和valid1后577维。physical248来自已审核current721[9:257]；0:9及257:721舍弃。
在线纯仿真host直接从实测刚体构造physical248，并做原生函数/参考扰动负控制。
启动采样窗口按Shadow8/60秒换算为4个Inspire控制拍，H4有效需4帧。
PPO的网络/优化完全一致；完整物体路径任务的奖励/成功需要有序进度与终点门控，
这属于任务目标适配，不冒充与Shadow静态终点任务奖励逐项相同。

数据：完整现有660序列（50物体）的同执行示教，总165677拍原生命令；
589 TRAIN/150044拍、71 DEV/15633拍，保留既有划分，DEV不进梯度/统计。
current BC使用全部有限命令，不要求未来24帧；WAM条件使用全部同刻可构造H4，
逐帧检查workspace，超出模型工作区的帧明确剔除scheduled训练并报告，不造τ。
每个序列都入当前BC；几何/已知物体路径与native来源一致。

预算：GPU6/7，既有monica低占用授权共享；GPU0–5的活动Shadow训练不改。
全量准备/BC阶段≤8小时、缓存+权重≤2GiB、可用空间<20GiB停止新增写入；
native工程PPO每次≤30分钟，训练若通过smoke遵从已有10000更新上限/48h并采用滚动保留。
不存视频、不重采660、不生成大token缓存。模型graspenv/Torch2.4.1+cu121，
物理dexplore_v120_train/Torch2.0.1+cu118；AF_UNIX有界RPC、临时文件在项目tmp。
