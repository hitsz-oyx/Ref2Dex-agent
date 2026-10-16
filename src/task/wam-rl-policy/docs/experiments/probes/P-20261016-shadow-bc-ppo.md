---
schema: ref2dex.probe.v2
probe_id: P-20261016-shadow-bc-ppo
experiment_id: P-20261016-shadow-bc-ppo
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-shadow-bc-ppo
probe_index_in_family: 1
seed_pool: probe
seeds: [216, 230, 231, 232]
decision_changed_if_positive: retain native BC-to-RL substrate and test frozen-WAM RL-token versus no-token and shuffled controls
decision_changed_if_negative: inspect residual exploration and task reward before investing in token adaptation
status: UNCLEAR
run_id: shadow-bc-ppo-20261016-r1
---

# Shadow冻结BC+小型PPO Stage-0

Class: Decision。服务Objective B的在线policy improvement接口。用户指定在wam-rl-policy
中推进并提供ref1路线：冻结WAM+BC，小RL actor/critic，先验证residual PPO机制。
此轮不声称验证WAM token效用，不更新1亿参数backbone，不同时添加learned token或新reward。

## 假设、决策与最低成本协议

假设：现有Shadow9k BC可作为在线探索起点，小PPO可在真实任务reward下改进保持；
正向则推进RLT式reference-conditioned/token对照，负向或UNCLEAR先排查探索与reward。
代码复用shadow-finetune的原生BC与pinned UniDexGrasp2 collector的初始化/physics，
24D canonical action，不套Inspire12D接口。BC/WAM eval且requires_grad=False，
optimizer仅新actor/critic，运行结束逐张量核对全部冻结state_dict未变且无gradient。

base=`outputs/shadow-finetune/shadow-expanded-20261015-r1/step9000.pt`，
cache=`outputs/shadow-finetune/policy-data-full-20261015-r1`。
RL observation=原生clamp300 +显式相对goal3/0.1m +BC reference24，总327D。
两个256D Tanh层actor/critic；actor均值末层全0，latent Normal初始log_std=-2.5。
动作`clamp(aBC+0.15*tanh(latent),-1,1)`；logprob与PPO ratio针对采样latent，
不把clamp后的动作当Gaussian。原官方goal_cond=False reward保持，sticky success只作指标。

工程门：PPO数学/终止/clip测试；BC模式与zero模式在原16env/panel/seed216下独立运行，
比较完整初态、canonical动作、force/torque/target与轨迹。初态不匹配不能比较后续。
模型初始化保存恢复所有环境RNG；探索使用独立CUDA generator232，不污染reset RNG。

通过后单seed230、32env（原panel两TRAIN物体各16），20个完整episode PPO更新，
每episode先原零动作warm reset，再199个60Hz动作，完整终点done全真/bootstrap0。
不做异步reset/短rollout截断；BC invalid/nonfinite直接停，不把零动作fallback当有效轨迹。
actor seed231，Adam1e-4；critic3e-4；gamma.99/lambda.95、4epoch、clip.2、jointKL .02
并rollback降LR、最大grad .5/1。BC正则.1×确定性执行动作偏差平方；不改变reward。

终点20更新不按DEV挑checkpoint；再以原固定16env/seed216独立BC/PPO配对评估。
主要连续60帧且goal<=5cm、相对初态lift>=10cm的保持代理，分TRAIN/DEV；同时读official
sticky与末帧5cm、reward、clipping、残差RMS、KL/更新次数。保持增加>=2/16且DEV不降
暂记PROMISING进入复核；下降>=2/16记UNPROMISING；其余UNCLEAR，不报总体成功率。
训练回报提升不能替代闭环确定性评估或接触稳定认证。

## 资源 / Decision Note

用户授权两线并行。Stage-0沿现有接口实施可逆最小改动，无新branch；0.15永久残差仅
第一轮接口，不把它确立为最终方法。GPU1，工程门<=600s/每臂，训练<=2400s，
终点评估<=600s/臂，总产物<=2GiB；Inspire最多GPU2/3/4，GPU0外部作业不动，6/7空闲。
边界：不改变MISSION claim，不增加预算，不触碰只读外部项目或原checkpoint。
超时、非有限、冻结/初态/执行审计失败停止，保留原始日志和完整episode边界checkpoint。
输出`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/`，启动后记录利用率/显存/ETA。

## Limitations / future evidence

单seed两TRAIN对象探索只检查机制；多对象/多seed、token aligned/shuffled/none对照、
reference dropout/独立动作路径及token训练留待正向信号后。训练checkpoint支持在episode
边界恢复网络/optimizer/RNG，物理环境须重新reset；snapshot不是完整PhysX checkpoint。
