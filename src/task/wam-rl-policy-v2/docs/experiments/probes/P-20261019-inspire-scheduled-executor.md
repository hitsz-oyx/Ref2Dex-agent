---
schema: ref2dex.probe.v2
probe_id: P-20261019-inspire-scheduled-executor
experiment_id: P-20261019-inspire-scheduled-executor
date: 2026-10-19
task: wam-rl-policy-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-scheduled-wam-executor
probe_index_in_family: 10
seed_pool: probe
seeds: [246]
decision_changed_if_positive: integrate the DExplore official PPO after the measured WAM BC bridge passes
decision_changed_if_negative: fix the first failing native or conditioning boundary before scaling
status: UNCLEAR
run_id: inspire-scheduled-20261019-r1
---

Decision工程Probe：能否将冻结WAMv4→空间goal scheduler→自训练BC执行器部署到
DExplore Inspire真实物理闭环。用户授权在Shadow评估后尝试，使用GPU6/7共享；0–5不动。
最低成本方案：一条TRAIN的s1_spheresmall_lift、300控制拍；先原生teacher正控制确认
该任务可执行，再用已有/本次同执行TRAIN示教拟合BC。只有通过32拍原生动作/PD审计，
才扩展到完整lift；之后再接官方DExplore PPO的有界更新，不启动大规模训练。
仅工程可行性，不把单序列拟合、teacher正控制或旧WAMv2结果作为新方法成功/泛化。

新合同：current delta0 721 + world displacement tau[8,11,3] + valid1=986，
30Hz H4[-3,-2,-1,0]，right slot，WAM点序输入/原生点序τ输出，单位米，
8未来时刻1/30..8/30秒；startup历史不足时tau=0/valid=0。
原生18动作中只训练12独立维度，六个coupling维度明确展开、由原生控制器覆盖。
冻结最终WAM80000、新Inspire canonical512/flow32/interaction128身份显式替换并绑定。
使用已知完整物体参考SE3路径，非未来实测手轨迹；当前原生721仍含同刻reference。

预算：单次900s、全流程≤3600s、≤2GiB输出、低于20GiB停止新增写入。
原生固定dexplore_v120_train(Torch2.0.1+cu118)，WAM/BC为graspenv(Torch2.4.1+cu121)，
有界本地AF_UNIX IPC，socket在项目tmp，不存视频/全量预测cache/多checkpoint。
原生执行复用collect_rl660_batch的初始化、capture和动作/PD审计，不更改vendor；
新adapter不得用官方teacher接管学生失败。记录动作、实测hand/object、原生contact、
终态/曾抬升与tracking，不能仅用物体z上升宣布稳定抓取。

### 正控制与首个分流

小球teacher300拍已完成，但最大抬升仅2.7mm、held0，未通过正控制；不把该任务失败
算作WAM负结果，也不拟合此失败示教。追加最多一个已知历史正控制TRAIN任务
s3_airplane_lift（542控制拍），若其teacher也失败则停止原生试跑并记录前提未满足。
这次任务选择修订先于BC拟合和学生评估，不扩展为物体搜索。

两次当前teacher失败后，停止对其做物体筛选和BC拟合。进一步只读比对发现，飞机的新旧
初始q/hand/object/current721/reference及整条reference_current全部相同，首个teacher
动作却已显著不同，权重文件仍一致；尚未定位是teacher future1442或推理接线差异。
为隔离这条Blocker，增加一次**已有TRAIN成功抬升canonical动作流的固定重放**（同一飞机，
不查询teacher、不换seed），验证本机物理任务仍可被已有动作执行；不作为reactive teacher
复现或WAM效果。如果重放实测held45，则可用已审核TRAIN原始示教先测试WAM条件BC；
若重放也失败，停止学生原生试跑。离线小BC只用原660已准入TRAIN数据，未准入失败新示教。

### 当前结果：接口通过，单序列BC未成功保持；PPO尚未执行

最终可复核入口：outputs/wam-rl-policy-v2/inspire-scheduled-20261019-r1/run-summary.json、
outcomes.json、measured-outcomes.png、fit/run.json、native-trial/process.json。
fit代码1845a300；原生学生运行代码13d8f983，原生peer按父启动器保存的命令/commit复现。
全流程未改GPU0–5的主PPO，也未加载官方actor作为学生权重或接管学生。

|条件|控制拍|最大抬升|最长保持代理帧数|最终保持45|动作捕获|
|---|---|---|---|---|---|
|当前reactive teacher小球|300|2.7mm|0|否|逐位一致|
|当前reactive teacher飞机|542|0mm|0|否|逐位一致|
|旧canonical动作流固定重放飞机|542|588.3mm|120|否|逐位一致|
|新WAM+BC自主飞机|542|61.4mm|0|否|逐位一致|

保持代理沿用既有partial_hold45：采样手物表面gap≤1cm、离开table支持、抬升≥3cm，
连续45拍（1.5秒）。不是接触力稳定性证书；重放后期失守且物体反复升降，不能称整任务成功。
首个replay因相对输入在native切换cwd后找不到文件而失败，修复路径resolve后以新输出r2运行，
失败产物保留；没有覆盖运行或checkpoint。

BC只用原660已审核TRAIN的s3_airplane_lift542拍（不是本次失败teacher轨迹），
3000全batch更新。WAM预测、数据准备与拟合共21.68秒，训练normalized MSE2.65e-5；
这是同一训练序列拟合，不是DEV表现。约5MiB fit产物（2.6MiB唯一BC权重），未生成全量缓存。
30Hz点序/时钟/anchor转换、新回合H4和独立动作合同测试通过；连同原Shadow相关合同共13测试通过。

smoke32的29个有效τ拍和完整542的539个有效τ拍全部为实际WAMv4预测，未喂未来实测手。
完整轨迹当前986初始输入与拟合输入逐位一致，所有发送canonical action与实际applied逐位相同，
输入/输出finite，source初始q/hand/object/current721一致。原生controller PD重建审计也通过。
这验证接口边界，不能据此证明预测轨迹可执行或WAM有效。
完整控制耗约107秒native窗口、模型worker122秒（含加载/等待），每拍模型处理平均88.7ms、
p95为111.3ms；仿真固定30Hz但wall-clock等待，不满足真实30Hz实时部署。

决策：单序列BC的自主抓取保持失败，不能宣称整套BC→PPO已经在Inspire成功。
本次PPO更新=0，没有移植或另写PPO算法；官方DExplore PPO adapter仍是下一步工程，
不把UniDex官方PPO直接冒充DExplore官方。应先补少量TRAIN多状态/多轨迹覆盖并接官方PPO
warmstart小面板，再讨论规模训练；不做全660大缓存或未经诊断的长训练。
teacher新旧首action差异仍未定位：旧/新current721一致不足以证明teacher delta1+16共1442一致；
只读调用链未发现新增super、restore或normalization确定错误，单列limitation，不误归因WAM。

验证说明：新增及相关接口13测试通过；本卡、seed和链接定向校验通过。全库verify仍因
trajectory-policy中6张既有实验卡缺v2元数据字段失败（39项），未修改这些无关记录。
全部试跑退出，GPU6/7恢复到原共享任务约349MiB/卡；总新增本次Inspire产物约17.1MiB。
