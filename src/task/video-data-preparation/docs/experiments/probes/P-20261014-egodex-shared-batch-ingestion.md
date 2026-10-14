---
schema: ref2dex.probe.v2
probe_id: P-20261014-egodex-shared-batch-ingestion
experiment_id: P-20261014-egodex-shared-batch-ingestion
date: 2026-10-14
task: video-data-preparation
branch: main
git_commit: 7558fb2c5e40e034ec08096729ccb616c9a32c22
claim_id: C1
hypothesis_family: HF-egodex-shared-track-admission
probe_index_in_family: 2
seed_pool: probe
seeds: []
decision_changed_if_positive: keep a bounded remote consumer producing auditable Shared-WAM windows
decision_changed_if_negative: fix the identified selector or processing blocker before expanding
status: UNCLEAR
run_id: egodex-shared-batch-20261014-r1
---

# 远端 Shared-WAM 批量接入

Result: UNCLEAR for automatic semantic admission. 首批50段/441窗完成，74数值候选；目标误选仍明显。r4抽查16窗批准5、拒绝10、延期1；两个VAL候选均误选。批量工程链路与Shared-WAM读取已跑通。
Decision: 保持现有几何门，用完整对象修饰词、透明物体延期和修订颜色门处理第二个不重叠50段cohort；全部自动候选仍须审查，不训练或全量扩量。

Class: Blocker/Decision. 用户明确授权停止一个远端 GPU 的既有任务并推进新线路批量接入。
保留 H4/K24、原生 2×11 手点、2048 scene、32 固定 object queries、观察/标签独立推理及
scale-only relative IQR≤0.25 等现有数值门。目标是检验自动目标 mask 能否支持可审查的真实
窗口，且队列消费、恢复、资源监督能实际运行；不训练、不形成方法有效性结论。

## Decision Note

- 关键证据：既有三窗 Probe 1 个数值合格、2 个标定拒绝；远端约18万完整配对已 READY，
  GPU0/1/2 为本任务辅助重建，GPU3 为主重建。新线路当前缺自动 mask 和外部监督。
- 行动：核验 GPU0 辅助进程671017 的 UID、命令与 GPU 对应，发送其支持的 SIGUSR1，
  让它完成当前 episode 后退出；保存停止和恢复信息。不动下载、主重建或 GPU1/2。
- 先复用已有模型资产和环境，在独立远端 scope 开发部署自动 mask backend 与受控 consumer。
  当前选择只读当前/H4；未来身份 mask 单独生成；自动候选的训练准入需可视化审查。
- 初次处理最多1 GPU小时、1000尝试窗口、新产物≤2GiB、scratch≤5GiB；新增模型资产≤5GiB。
  先查看20–50个简单刚体 episode 的合格率/拒绝归因，再决定同资源内下一批；外部 owned
  watchdog 限时，模型失败、陌生GPU进程、空间不足或预算触发即停止。
- 用户已授权这次 GPU 切换及批量接入；无新分支、无未知进程操作、无覆盖旧产物。
  未解决的选择歧义进入拒绝/待审队列，不以放宽几何质量门提高通过率。

## Identity / outputs

实现起点为上列提交；实际执行提交将在启动前固定并写入 manifest。
远端独立目录：`/mnt/ugreen_nas/storage/oyx_storage/egodex-shared-batch-20261014-r1/`。
本地证据：`outputs/video-data-preparation/egodex-shared-batch-20261014-r1/`。
原始下载与旧重建只读，aux 停止仅发生在明确授权且核验过的 owned process。

## Limitations / future evidence

自动语言定位和表面分割是伪标签；数值门不能证明目标语义正确或米制真值。
先生成审查产物与明确准入记录，再冻结可训练 train/dev；不承诺全量覆盖或全量完成时间。
独立几何精度、多 session 泛化与训练收益延后。

## 已完成部署

GPU0旧worker671017收到SIGUSR1后在当前episode结束正常退出，状态STOPPED；其余三个重建
worker和下载producer保持运行。停止请求及完整恢复命令保存在remote scope的
`old-worker-stop-request.json`，没有强杀或删除旧数据。

执行代码 `7dfcdec2`，独立解包为remote `repo-7dfcdec2/`；首批清单
`ready-cohort-e9a9fa72.jsonl` 固定50 episode/37 session。原session split registry继续复用，
未为凑dev重新分组。模型资产：Grounding DINO tiny revision `a2bb814`，SAM2.1 tiny，
SAM2源码commit `2b90b9f`；Omega/CoTracker复用远端既有资产。
缺失iopath/portalocker只安装在scope-local `packages/`，没有修改生产conda环境。

工程grounding在真实积木当前帧完成11.28秒、峰值1.91GiB；它暴露了多积木候选歧义，
因此selector明确加入已观测H4手运动和当前手邻近prior，仍不查看未来运动。语言解析也
修复了把不支持对象的来源容器误选为target的问题；自动结果仍保持待审，不当作语义真值。

Supervisor PID1557492，remote `batch-r1/` 保存launch、resources、consumer.log、corpus；
外部deadline3900秒（含初始化），另有最多90秒退出宽限；consumer处理限3600秒/1000窗口/2GiB，scratch5GiB。
停止入口为该scope的STOP文件或具名supervisor SIGTERM，只处理它自己的child session。
准备阶段48项测试通过；后续当前selector/审查promotion测试3项通过。

## 首批检查与局部修复

r1处理到152尝试窗口时，26个数值合格候选，GPU显存约10289MiB，耗时412秒；候选
保持不可训练。首个9窗四帧RGB审查中，7个目标/轨迹未见明显错误，2个把绿色积木认作
白色积木，已拒绝。7个批准窗口复制到独立 `approved-r1/`，TRAIN7/VAL0，仍不能启动训练。
原候选和审查图保持不变；本轮没有证明绝对米制几何或完整中间时序正确。

Decision Note: Grounding DINO对颜色的约束不足是已观察到的selector缺陷。追加当前可见
mask的保守颜色矛盾屏查（允许白物体的暖色照明），保存原始target/hand PNG，以便审查。
两个已知绿色误选在CPU中心区域诊断中均被拒绝，真实白色窗口未被这一局部诊断拒绝；
该诊断不是完整SAM mask验证。准备按新代码在独立r2 scope复跑，保留全部r1证据；
累计GPU处理仍限制在原1小时预算内，不修改0.25尺度门或其他几何门。

用户目视确认 `review-r1/candidate-004.jpg` 跟踪良好，`candidate-003.jpg` 确实误选目标。
两窗确认另存 `user-visual-confirmation.json`；不将这两窗反馈扩为全语料准确率。

r1正常停止耗时538.98秒。新代码7558fb2c的r2在初始化阶段因本任务新下载的
`preprocessor_config.json` 变为mode000而失败，尚未进入GPU推理；同目录配置和权重
也不可读。最小复现为直接读取该JSON，PermissionError；对这一确切文件恢复0644后
相同读取成功。NAS映射uid1005/gid10与执行uid1006不同；已有下载器显式chmod0644，
但NAS权限变化的根因未确认。只恢复本任务asset manifest列出的新文件，不修改原始数据、
旧教师资产或系统权限；后续新资产发布补显式0644。保留r2失败日志，恢复后在新scope重启。

r3进一步暴露同类权限问题：此前直接写入的 `session-splits.json` 不可读，32.12秒在
初始化退出；这是同一NAS输入发布问题，不是selector或标定负证据。修复该确切清单后，
对分组清单、cohort、三个模型checkpoint及grounding模型文件共13个必要输入读取检查
全部通过；已有corpus的JSON/HDF5仍可读。没有对整个共享目录批量修改权限。

恢复运行r4：同一颜色筛查代码7558fb2c，Supervisor1570268/consumer1570269，scope
`batch-r4/`，consumer最多2700秒，外部deadline2820秒加最多90秒退出宽限。所有旧运行
加grounding smoke累计588.15秒；最坏情况下总计仍小于1GPU小时。修复NAS输入发布及
将split metadata提前检查的后续工程提交为f62ef748，不冒充已经运行的7558fb2c版本。

独立批准语料 `approved-r1/` 经真实Shared-WAM provider/collate读入4个窗口，scene
`[4,2048,3]`、features`[4,2048,18]`、scene labels`[4,1,24,32,3]`、hand labels
`[4,24,2,11,3]`，float全部有限、source_id均3。报告
`outputs/video-data-preparation/egodex-shared-batch-20261014-r1/approved-r1-shared-adapter-audit.json`。

## r4稳定运行与CPU追加审查

必要输入修复后，r4已实际产出窗口。153尝试/26候选，417.97秒，GPU0约10289MiB，
其余下载和三个重建worker仍在运行。首批cohort45 TRAIN/5 VAL，441个候选anchor；
当前较早的VAL episode未产出数值候选，后续4个VAL episode仍待消费，不调整原split。

完整当前SAM mask进一步检查揭示7558fb2c的初始颜色门较弱：两个绿色误选的旧矛盾分数
仅0.452/0.494，仍会留下待审候选。更新颜色条件的CPU重算为0.827/0.852，真正白色
窗口为0.0。对应修复/审查工具提交aa1f625a；r4运行代码仍固定7558fb2c，不热改它的
模型或标签。用保存的current target/hand PNG和原始当前RGB作CPU审查补充；没有重新
选择query、改变未来标签或自动批准未审窗口。该三窗检查只能支持局部修复，不能声称
颜色语义在全数据上可靠。后续backend运行可采用新门，本轮当前候选仍需显式审查。

用户说明本任务优先，若需更多GPU可以提出。目前首批一张GPU足够；优先获取selector
和train/dev准入信息，未申请或抢占额外GPU。完整训练、任意任务自动grounding和全量
覆盖仍不属于当前首批完成状态。

## 首个cohort完成与下一cohort

r4正常DRAINED：50 episode，441尝试，74数值候选（TRAIN72/VAL2），30 episode产生
分片、20 episode无合格窗口；处理1127.44秒，GPU峰值约10.05GiB，corpus约9MiB。
0.25尺度门未变：91窗UNCALIBRATED；其余主要拒绝为endpoint89、observed/label depth61、
当前歧义43、刚体/噪声40。数值候选率16.78%，不能解释为语义准确率。

CPU审查r4的16窗：5批准、10拒绝、1延期。误选包括绿色/静止块、干燥架和透明瓶子的
邻近蓝色物体；两个VAL候选分别选中另一个蓝色物体和红色玩具，均不批准。不得为了凑dev
修改split或放宽目标身份。r4批准语料独立 `approved-r4/` 为TRAIN5/VAL0；r1的TRAIN7
是不同profile的旧证据，不合并重复窗口，不相加当成12个训练样本。

语言解析保留完整名词修饰，避免把“yellow rhombus block”降成泛指“block”；明确延期
clear/translucent物体。当前语义定位仍需进一步审查，不以GPU数目替代这个blocker。
第二cohort用 `--exclude-manifest` 排除前50个episode，50段/36 session。执行代码30b1e9a2，
remote `repo-30b1e9a2/`、`config-30b1e9a2-r5.json`、`ready-cohort-next.jsonl`。
新scope `batch-r5/`；Supervisor1588113，consumer PID在launch.json记录。consumer1500秒，
外部deadline1620秒加90秒退出宽限；先前总计1734.60秒，最坏总计3444.60秒，仍在原
1GPU小时预算内。不使用第二张GPU、不修改现有下载和其他三个重建进程。

第二批正在运行时本卡status保持RUNNING；工程结果已保留，尚无可训练的非空dev快照。
停止/预算触发保留已发布分片；后续扩量应基于语义审查结果，使用新运行预算记录。

## r5正常完成与最新只读状态核验

2026-10-14 03:36:40 UTC（北京时间11:36）现场核验：r5 supervisor为COMPLETED/CHILD_EXIT，
returncode0，1275.23秒；consumer为DRAINED，50段全部处理（PUBLISHED32/REJECTED18），
481尝试、122数值候选、training_admitted0，处理计时1240.80秒，峰值10289MiB。
主要拒绝仍是scale90、endpoint84、观察/标签depth58及当前歧义38，几何门未放宽。
累计含此前1734.60秒共约3009.83秒，未超原1GPU小时预算。GPU0当前空闲；本轮未重启consumer。

下载producer399615仍RUNNING：completed_packets189064，part1完成、part2剩121失败配对、
part3下载中。旧主重建360908和辅助671018/671019进程仍存在，GPU1/2/3未被本轮修改。
这些旧重建属于既有线路，不能称为全部已通过本Task的Shared-WAM准入。
证据：本地输出根下`latest-remote-status.json`、`r5-completion-summary.json`。

当前自动语义准入判断仍UNCLEAR，r5候选待视觉审查，无训练启动。用户转向UniDexGrasp小样本
核验，本轮不继续增加远端GPU批次；下载和旧重建继续自主运行。后续若恢复新批次，先做目标
审查与新预算记录，不能把数值候选直接当训练数据。
