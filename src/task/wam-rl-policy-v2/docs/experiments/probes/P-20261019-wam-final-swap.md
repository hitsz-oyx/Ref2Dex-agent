---
schema: ref2dex.probe.v2
probe_id: P-20261019-wam-final-swap
experiment_id: P-20261019-wam-final-swap
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 6
seed_pool: probe
seeds: [246]
decision_changed_if_positive: consider adapting executor training to the final WAM checkpoint
decision_changed_if_negative: retain old WAM and diagnose trajectory distribution shift
status: COMPLETED
run_id: wam-final-swap-20261019-r3
---

Decision：固定PPO5500轮，比较旧WAM12000、新contractfix最终80000与DEV best30000，
判断直接替换是否改善闭环表现。GPU6/7，每条件相同16 TRAIN物体×4环境，
seed246，确定性动作，预热reset后重设seed再次reset，首个200步native回合。
每进程900秒，不更新PPO、不改主训练。graspenv；沿用已有评估协议与成功/保持指标。
新checkpoint导出model/config/stats/schema/step约400MB/份，不复制优化器；
远端已完成run为pretrain-nine-v4-contractfix-warm13000-20261018-r1，
另一个scale1024运行尚未完成，不混为本次最终模型。
新模型绑定dense manifest摘要1e2ec0cdba3eca9825232bdf2df45db9401fbba39482038c099fbcbbc14e53f7，
旧本地manifest不同；须提供匹配文件及几何payload，不关闭身份校验。
当前远端bound manifest权限000、wbcd无法读取，已请求用户提供可读副本/恢复权限。
产物目录outputs/wam-rl-policy-v2/wam-final-swap-20261019-r1/。

旧WAM对照已完成：PPO5500、64场景，54/64官方成功（84.4%），39/64保持（60.9%），
workspace_failures=0，optimizer_steps=0。与此前同配置结果存在几例波动，
本次应使用同期对照，不以历史79.7%作为固定真值。新模型未评估：bound manifest
权限000，需要所有者提供可读副本。权重正在下载；完成后后台CPU加载验证写入
outputs/ptv3-shared-wam-v4/local-control-final-20261019-r1/download-status.json。

用户随后明确授权仅替换checkpoint、沿用现有本地geometry。新增仅评估可用的
--allow-wam-geometry-replacement，显式记录训练绑定摘要与实际manifest摘要，
不修改checkpoint，不关闭本地payload/32query/512surface有效性检查。
结果仅解释为“固定本地几何下直接换权重”，不是严格复现远端几何的部署。
有效run_id=wam-final-swap-20261019-r2；同时重跑旧权重对照。

## 直接换权重结果（2026-10-19）

用户确认新模型为既有WAMv4训练完成后的同接口checkpoint，授权沿用本地几何。
两个模型精简导出各395,075,977字节，564 tensors，完整CPU加载通过。
保留model/config/stats/schema/step；不修改权重或归一化。运行代码18ed7b69，
有效比较run为wam-final-swap-20261019-r3，PPO固定5500，seed246，64首回合。

|WAM|官方成功|抬高10cm且目标5cm内保持60步|平均回报|
|---|---|---|---|
|旧12000|46/64 (71.9%)|35/64 (54.7%)|131.09|
|最终80000|51/64 (79.7%)|42/64 (65.6%)|146.71|

两者初始565观测、hand、object pose、goal逐位一致；完整root状态并非完全一致，
差异仅table actor：位置最大7.63e-6m、线速度0.00866m/s、角速度0.01305rad/s。
因此是有残余物理扰动的小样本Probe，不是严格完整状态配对验证或显著性结论。
成功增加5个、保持增加7个，支持继续评估/适配新WAM，不证明泛化。
30000 best诊断41/64成功、29/64保持；初始手物状态未匹配，排除于主比较，
不能据此判断DEV best劣于final。此前r1/r2重复亦仅diagnostic，不混合汇总。

评估初始化额外明确清除reset_goal_buf（否则预热后条件分支额外消耗随机数），
并在reset前清除桌面速度；完整物理重复性仍有上述边界，未证明全部解决。
全部三组模型严格state_dict加载成功、optimizer_steps=0、workspace_failures=0。
36项既有测试通过，新增替换模式仅评估入口开放；新权重用本地manifest，
payload与query身份检查保留，训练绑定摘要与实际manifest摘要均记录。
主训练r4未更换WAM；GPU6/7释放。后续优先复用这个80000权重做更广评估，
不要依据单次64场景直接宣称收敛改善。
