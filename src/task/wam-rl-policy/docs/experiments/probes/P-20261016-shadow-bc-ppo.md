---
schema: ref2dex.probe.v2
probe_id: P-20261016-shadow-bc-ppo
experiment_id: P-20261016-shadow-bc-ppo
date: 2026-10-16
task: wam-rl-policy
branch: main
git_commit: 2879b585
claim_id: C1
hypothesis_family: HF-shadow-bc-ppo
probe_index_in_family: 1
seed_pool: probe
seeds: [216, 230, 231, 232, 233, 234]
decision_changed_if_positive: retain native BC-to-RL substrate and test frozen-WAM RL-token versus no-token and shuffled controls
decision_changed_if_negative: inspect residual exploration and task reward before investing in token adaptation
status: UNCLEAR
run_id: shadow-bc-ppo-20261016-r1
---

# Shadow冻结BC+小型PPO Stage-0

Result: UNCLEAR。真实原生PPO20更新完成，冻结/执行审计PASS；三对固定终点reset面板
保持11/48→15/48，但方向+3/-1/+2不一致且BC自身回放波动显著，不足以确认RL收益。
Decision: 保留Stage-0接口，优先检查有效探索、饱和与reward/保持关系，再决定扩大
在线预算或接learned RL-token；不更新WAM，不把本轮当作token效用或正式RL改进证据。

Class: Decision。服务Objective B的在线policy improvement接口。用户指定在wam-rl-policy
中推进并提供ref1路线：冻结WAM+BC，小RL actor/critic，先验证residual PPO机制。
此轮不声称验证WAM token效用，不更新1亿参数backbone，不同时添加learned token或新reward。

## 固定终点实际读出

20更新、127360环境步、约1149s；80个actor epoch均接受，jointKL最高.017618，
behavior latent logprob最大误差5.73e-6。597个冻结字段逐值未变且无gradient。
逐拍canonical/yaw、force、torque误差均0，finger-target最大1.2e-7。

| reset seed | BC保持/16 | PPO保持/16 | BC DEV保持/8 | PPO DEV保持/8 | official BC/PPO |
|---|---:|---:|---:|---:|---:|
| 216 | 5 | 8 | 1 | 1 | 11/11 |
| 233 | 4 | 3 | 0 | 0 | 8/9 |
| 234 | 2 | 4 | 0 | 2 | 7/6 |

合计保持11/48→15/48（TRAIN10/24→12/24、DEV1/24→3/24），official26/48→26/48，
末帧21/48→22/48。三对全部九类初态逐值相同，独立从raw状态复算各split指标一致。
PPO确定性执行残差RMS .00233–.00249、intent clipping约29%–31%；训练采样执行RMS约.01、
裁剪35%。这些表示探索/修正已实际进入执行，不是成功率或残差足够大的证明。

训练前5/后5episode回报均值131.38/126.27，reset不同，仅说明曲线未现清晰上升，
不把它视作配对退步证据。保持方向混合且BC自身回放波动明显，最终UNCLEAR。
训练commit2879b585，评估dcce7a33；模型train/latest.pt、评估evaluation-pairs.json、
汇总summary/results.json/csv与comparison.png，均在run根目录。六次冻结/执行审计PASS。
全部运行完成退出，GPU1–7释放，不动GPU0外部作业。

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

后续官方 initialized PPO 对照显示，已有 native policy 在官方网络和 reward 配方下可
保持抓取，而新增 WAM 通路会迅速破坏该能力；这不改变本卡小面板 frozen-BC residual
PPO 的 `UNCLEAR` 状态，也不能把本卡的少量行为变化解释成官方 PPO 或 WAM 的收益。

## 资源 / Decision Note

用户授权两线并行。Stage-0沿现有接口实施可逆最小改动，无新branch；0.15永久残差仅
第一轮接口，不把它确立为最终方法。GPU1，工程门<=600s/每臂，训练<=2400s，
终点评估<=600s/臂，总产物<=2GiB；Inspire最多GPU2/3/4/5，GPU0外部作业不动，6/7空闲。
边界：不改变MISSION claim，不增加预算，不触碰只读外部项目或原checkpoint。
超时、非有限、冻结/初态/执行审计失败停止，保留原始日志和完整episode边界checkpoint。
输出`outputs/wam-rl-policy/shadow-bc-ppo-20261016-r1/`，启动后记录利用率/显存/ETA。

## Limitations / future evidence

2026-10-16用户要求优先看Shadow表现：不增加训练预算，从全部已存轨迹做行为分层与
交互式回放。每物体12次，BC/PPO曾保持分别瓶子4/5、相机6/7、手机0/1、书1/2。
抬升10cm总数均29/48、到过目标均26/48；末段持续60帧到结束为11/48→14/48，低于
曾保持11→15。PPO仍15/24 DEV未抬高10cm；净增保持包含7条改善和3条退步，不是普遍提升。
相机seed233/env6曾保持114帧，终点lift11.49cm/goal距离9.26cm；不把离开目标区自动说成掉落。
行为总表、全部case CSV、原始日志点云/11关键点HTML回放和对象统计图位于run下
`behavior-review-verified/`。CAD identity/scale与日志有限性校验通过；Node检查全部48case
各5帧canvas坐标有限，统计PNG已查看，但尚无真实浏览器视觉检查。无新仿真或网络计算，
纯文件统计使用CPU。原单训练seed/小面板/非接触认证/数值波动等限制全部保留，结论UNCLEAR。

单seed两TRAIN对象探索只检查机制；多对象/多seed、token aligned/shuffled/none对照、
reference dropout/独立动作路径及token训练留待正向信号后。训练checkpoint保存未来恢复
所需的网络/optimizer/RNG，当前CLI仅evaluate加载actor，尚未实现train-resume。
物理环境须重新reset；snapshot不是完整PhysX checkpoint。

## 工程门修订（PPO启动前）

原BC/zero两个独立进程初态各字段完全相同，但完整闭环轨迹不相等，原精确轨迹门FAIL保留。
独立原shadow-finetune执行器也出现首拍约8.4e-6的command偏差。最小复现固定同一初始
16env输入、冻结权重、每组8次推理：free/fixed RNG、no_grad/inference_mode均有
6.1e-6至1.12e-5的重复command差异；固定RNG未消除。每次same-packet BC+0逐值完全相等，
冻结参数未变。证据在reference-repeatability.json；不把完整轨迹差异当RL提升或接口失效。

Decision Note：完整轨迹逐值门在既有GPU稀疏推理数值波动下不可用。改用相同输入的
same-packet零残差逐值一致、初态逐值一致，以及逐拍独立解析yaw旋转/force/torque/
finger-target执行核验。原失败不覆盖、不调宽轨迹tol。推进Stage-0机制Probe，但单次
BC/PPO结果须结合BC自身回放波动，仅描述、记UNCLEAR，明显正信号也先做重复回放确认。
固定终点保持指标与训练预算不变；不是科研判定门的事后放宽。

终点评估追加（查看PPO终点行为前记录）：等价BC的3/16波动已超过原±2/16门，不能再
以一次±2/16判断RL正负。保留216为原主面板，另预先固定233/234两个reset seed，
每seed在同GPU独立BC/PPO配对各16env，核对全部初态。六次回放每臂总预算仍<=600s，
三对可并行，输出eval-seed*/。三对方向冲突或增益处于已知BC波动范围内均UNCLEAR；
只有方向一致且明显超出波动才支持进一步重复，不升级正式方法结论。
