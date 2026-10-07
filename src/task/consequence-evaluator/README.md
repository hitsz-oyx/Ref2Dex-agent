# Consequence evaluator

用户方案：[ref1](docs/user/ref/ref1.md)，当前合同按[ref2](docs/user/ref/ref2.md)修订；数据分级按[ref3](docs/user/ref/ref3.md)。实现分支：`consequence-evaluator`。
当前工作只建立离线 oracle headroom 检验，不改变根级 Mission 的最终 Cm 策略收益要求。

第一阶段固定 `K=24`、`K_exec=8`，H 沿用采集策略当前观测/历史的原始合同。
先采连续六专家 rollout，单次24步平滑扰动后让专家继续到 episode 结束；不 fork。
比较 `E0(H,δ)`、`Eoracle-E(H,δ,Z_object)` 和 `Eoracle-EI(H,δ,Z_object,Z_interaction)`。
δ为执行前已确定的24步18维请求残差计划，专家仍逐步反馈控制；PointWorld 适配与在线 proposal 延后。
主要衡量同任务、同当前阶段、跨 episode 的局部偏好排序；它不是同状态候选控制收益。

## 已实现的最小合同与训练入口

- [窗口准备](tools/run/prepare_windows.py)：episode/seed group 先 split，再切完整24步窗口。
- [连续采集](tools/run/collect_continuous.py)：复用原生六专家player，分阶段单次24步平滑扰动，完整episode导出。
- [数据合同](src/consequence_evaluator/data.py)：严格输入白名单、完整时钟、刚体效应、标签 mask 与 split 检查。
- [三臂模型](src/consequence_evaluator/model.py)：同容量三臂两层128维 Transformer，24步 progress 分布与独立标量分支评分。
- [训练](tools/run/train_matched.py)：相同初始化、抽样、优化器和更新预算；只在训练集拟合归一化，用 val 保存最佳权重。
- [独立评价](tools/run/evaluate_matched.py)：冻结val所选checkpoint后，报告test排序、任务/阶段分组及跨episode未来替换对照。
- [资产预检](tools/audit/preflight.py)：核对六专家、motion、旧 rollout 和当前 GPU 占用。
- [研究原文](docs/research/PRIMARY_SOURCES.md)与[代码/数据复用核对](docs/DATA_REUSE.md)。

三臂保留相同45维 future module 和初始化。归一化后，E0屏蔽全部future，
Eoracle-E仅保留12维物体effect，Eoracle-EI再保留11个测量手关键点的相对物体坐标（33维）。
模型输入只有H、执行前已知δ、物理future，不接收未来reward、success、drop/contact flag、
episode quality、扰动身份或时间到成功。物体effect为 `T_t^-1 @ T_(t+1:t+24)`，
手部future使用每个未来物体坐标系；具体顺序由 `contracts.HAND_LINKS` 固定。
请求残差区别于裁剪/噪声后的actual_residual；后者与真实闭环action只用于审计。
未来状态触发起点尚未知的窗口被排除，clean为已知零计划，触发后剩余计划已知。
18维残差接口与PointWorld的手部point-flow仍需适配；目前PointWorld只预测物体effect，
尚不能声称恢复interaction oracle的差距。

Robometer 原版使用联合两轨迹 preference head；独立 `s` 的 Bradley–Terry loss 是明确改编。
Progress 用10-bin soft CE，只监督可靠 `expert_success` 的绝对 episode 进度，失败/次优 mask。
不把每个窗口重新标成0→1，也不把 episode 最终失败直接继承为全部局部窗口失败。
首版要求可审计的局部偏好 annotation；不自动产生旧Y或从窗口外结局伪造局部排序。
[监督准备](tools/run/label_continuous.py)只对明确的保持/掉落和抬升/失抓窗口给出偏好，
模糊的miss/recovery比较不标。可靠clean轨迹的45帧保持且之后不掉落用于绝对progress锚点；
perturbed/失败/次优精确progress均mask。三臂另抽相同的train专家窗口，保证progress锚点不依赖偏好配对覆盖。

### 当前 fit gate

局部偏好必须同时匹配历史 `H`、当前物体位姿和11点手状态。`H` 的相对RMS上限为
25%，物体/手的几何阈值仍按上面的固定合同执行。每个 split 还必须有至少
8/4/4 个不同的无序 episode pair（train/val/test）；同一对 episode 的重复窗口不能
充数。collector 只把该要求写入来源 manifest；labeler、prepare、`Windows` 和 train
会实际核对配对数量以及当前规则、合同和来源 hash。六个专家全部通过固定资格并且
route 的 `training_allowed=true` 之前，连续采集和真实 fit 都保持 observational-only。

同一 current state 的双分支由 `src/consequence_evaluator/twin.py` 定义
`ref2dex.consequence-evaluator.twin.v1` 合同：必须保存完整 native task/controller
buffer、Python/NumPy/Torch RNG、fresh simulator prefix replay provenance 和双分支第0帧
锚点。当前已提供不导入 Isaac 的 native capture adapter，可把初始化后的 task、controller
buffer 和完整 prefix trace 转为该合同；它会区分 30 Hz control tick 与原生 60 Hz
physics frame（按 `control_freq_inv` 核对帧数）、拒绝缺失字段、非 fresh replay、状态帧数不匹配
或超出 replay 误差门槛的记录，并冻结 task 的直接 tensor/scalar inventory、禁用随机化和
motion sampler。通用 snapshot 要求显式 `{'is_rnn': ..., 'state': ...}` sentinel，并校验
Python/NumPy/Torch RNG 的具体格式，并提供对应的 CPU/GPU Torch RNG restore helper。
native DExplore adapter 进一步固定 `control_freq_inv=2`、`sim_params.dt=1/60`；PhysX
`substeps` 必须由 caller 放入 physics provenance hash，但不计入 Gym frame count。
adapter 尚未接入连续 native collector 的双分支执行，
也没有真实 twin branch 产物；非RNN controller 也必须显式记录空 RNN sentinel。合同测试不能
作为 twin coverage 或 evaluator 科学证据。

## 连续 episode 输入

源目录的 `manifest.json`：

```json
{
  "schema": "ref2dex.consequence-evaluator.episodes.v2",
  "status": "COMPLETED", "rollout_kind": "continuous", "training_allowed": true,
  "fps": 30, "units": "m",
  "action_semantics": "decision_known_requested_residual_plan",
  "history_contract": "采集策略的观测/历史 schema、维度与来源 SHA256",
  "episodes": [{
    "episode": "unique_episode_id", "split_group": "source_seed_group",
    "split": "train", "task": "airplane", "quality": "expert_success",
    "expert": "airplane_base", "motion": "s3_airplane_lift",
    "path": "unique_episode_id.npz", "sha256": "实际文件 SHA256"
  }]
}
```

每个 episode NPZ 使用 `allow_pickle=False`，仅含以下字段：

| 字段 | Shape / 含义 |
| --- | --- |
| history | `[T+1, ...]`，策略原始观测/历史，不新定义history length |
| action | `[T,18]`，真实执行的[-1,1] native control，仅作审计 |
| residual_plan | `[T,24,18]`，每次执行前已知的请求残差，模型action取此字段 |
| plan_known | `[T]`，bool；尚未知未来触发时刻的计划不可训练 |
| hand_keypoints | `[T+1,11,3]`，测量native刚体关键点世界位置 |
| object_pose | `[T+1,4,4]`，测得的 object-to-stationary-world，米 |
| timestamps | `[T+1]`，真实连续30Hz |
| phase | `[T]`，当前状态决定的采集阶段，不作为额外模型输入 |
| progress | `[T+1]`，可靠成功示范的绝对进度；未知可NaN |
| progress_mask | `[T+1]`，bool；失败/次优全部false |

其中 `action[t]` 把 `object_pose[t]` 变成 `object_pose[t+1]`。
episode 文件必须完整，不能把 reset 后的另一条轨迹拼进来。
采集驱动已经接入，使用CUDA PhysX tensor pipeline；真实六专家连续采集仍待新权重资格检查后执行。
CPU测试实际执行驱动中的采集循环，覆盖不同env先后done、原生in-place action转换、
零partial reset、T控制/T+1状态对齐和恢复专家控制；它不能证明真实物理响应或阶段质量。

采集模板（权重必须匹配路由配置SHA；本地native仿真资产和配置也须就绪）：

```bash
python src/task/consequence-evaluator/tools/run/collect_continuous.py \
  --route-config src/task/CmResidual/configs/multitrajectory_object_router_with_cup_probe.json \
  --asset-root <restored-checkpoint-root> --motions <native-motion-directory> \
  --cfg-env <native-environment.yaml> --cfg-train <native-player.yaml> \
  --output outputs/consequence-evaluator/<continuous-run> \
  --gpu <idle-gpu> --seed <registered-collection-seed> --split train
```

默认24env/2waves/每episode最多1200步（覆盖最长1062帧参考）/总900秒/产物2GiB。开始前核对输入hash、GPU PID与20GiB磁盘余量。
六专家始终用原生RMS和动作接口；可传入原观测路由模型及SHA，否则使用明确记录的固定object route。
每episode分配clean或approach/contact/grasp/lift/hold，阶段达到且剩余至少24步才启动一次残差。
平滑残差使用四结点插值和零边界taper，腕平移缩小至四分之一，native coupling覆盖的6个distal通道不加残差。
阶段定义是当前contact force proxy、3步contact、3cm lift与5步held的工程启发式，不能称为论文原版阶段。
目标阶段未达到的episode保留未触发状态，不计作扰动覆盖；完成一段扰动后完全恢复专家。
原生实现的`hybridInitProb=1`选择参考第0帧；运行时强制并核对完整起始帧。不同episode只在wave边界统一reset。
在sidecar记录真实q、hand root、contact及validity、实际残差/裁剪，供后续几何适配与标签审查；contact/force/gap、q/root与残差审计不进入Z；独立测量手关键点进入interaction future。
审计用执行动作从原生`pre_physics_step`入口捕获：wrapper裁剪及domain action noise已执行，Inspire手指/PD转换尚未执行。
不把请求的动作冒充实际指令；原生噪声若使真实归一化指令越过[-1,1]，合同检查直接拒绝该运行。
reset瞬间的contact force可能残留，标为无效，不用于阶段计数。采集progress保持未知且全mask，
不会根据最终结局自动生成局部preference。可靠专家progress与窗口内事件排序必须在独立监督步骤核验后提供。

偏好文件单独记录 `scope: local_window`、`label_provenance`，以及 `pairs`。
每对包含 `chosen` / `rejected` 的 `{episode, tick}` 与 `annotation`。
标签来源必须核对窗口内的具体事件；无明确局部顺序的比较不标。
程序验证 split/task/expert/motion/phase 相同、episode不同、annotation非空，
当前物体平移≤3cm、z差≤1cm、旋转≤15°，手11点RMS≤2cm；仍是observational matching。
事件标签要求净力proxy同时有≤1cm采样手物表面距离；稳定progress再要求抬升≥3cm、
持续45帧且之后不掉落。净力proxy及采样距离均不是精确collision-pair GT，
扩大采集前须检查真实轨迹的force/geometry一致率。

准备后运行的模板（真实源、偏好审计、空闲GPU和注册seed就绪后）：

```bash
python src/task/consequence-evaluator/tools/run/prepare_windows.py \
  --source <continuous-episode-directory> --preferences <local-preferences.json> \
  --output outputs/consequence-evaluator/<prepared-run>
python src/task/consequence-evaluator/tools/run/train_matched.py \
  --data outputs/consequence-evaluator/<prepared-run> \
  --output outputs/consequence-evaluator/<fit-run> \
  --gpu <idle-gpu> --seed <registered-probe-seed> --updates 1000 --batch 32 --seconds 1800
```

训练不使用 test pairs；训练入口只报告 val开发指标。保存三臂 initial/latest/best、训练/验证JSONL、
输入/源码hash、显存、吞吐与ETA。当前入口尚未在真实GPU数据上验证，CPU小模型检查只证明工程合同。

独立评价入口要求训练已经COMPLETED，核对原始输入/初始化/源码hash、train-only归一化和
验证日志中首次最佳的checkpoint。三臂采用相同选择规则，可以选中不同更新步；不会根据test重选。
第一次推理前在fit目录冻结`test_protocol.json`，之后不能通过换权重、诊断seed或batch重扫test。
保存每个参与test偏好对的唯一窗口分数/progress、偏好对索引和未来donor索引，便于独立复算。

```bash
python src/task/consequence-evaluator/tools/run/evaluate_matched.py \
  --data outputs/consequence-evaluator/<prepared-run> \
  --fit outputs/consequence-evaluator/<fit-run> \
  --output outputs/consequence-evaluator/<test-run> \
  --gpu <idle-gpu> --seed <fixed-diagnostic-seed>
```

主指标是test strict preference accuracy，tie计错；报告配对gain、三臂救回/损失的偏好对数量、
task/phase/quality分组和episode-pair macro accuracy。Progress MAE只统计明确可靠的masked帧，
同一窗口在多对出现不重复计算。另用相同oracle权重替换Z：同task/expert/motion/当前phase、相近当前物体/手状态、其他test episode
有放回抽取，H/A/labels均保持原样。这是未来对齐敏感性的诊断，不是严格置换检验或可执行候选。
窗口重叠，不把pair数量当独立样本数；当前汇总不提供独立pair CI，也不自动升级Validation。
默认最多128窗口/batch、300秒；GPU占用检查在加载模型前完成。31项微型CPU测试通过，
原因是当前GPU均占用，测试只验证数据合同/保存加载/评价逻辑；真实GPU采集、拟合与评价仍未运行。

## 当前判断与下一步

用户在2026-10-07明确允许重新训练专家并重新rollout。已找到外部read-only的canonical
几何motion和native资产，13条motion完成CPU合同检查。资产复制到
`outputs/consequence-evaluator/baseline-inputs-20261007-r2/`；原DExplore data失效链接已保留。
这只恢复训练输入，不恢复旧checkpoint或robot rollout。历史s3直接scratch曾失败，先新训s1 parent
并做完整frame0检查，再迁移s3、mixed12/train5/balanced5及duck/cup，最终生成新六专家route/hash。
[重建协议](docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md)保留成本、资格检查和下一步。

2026-10-07：旧 oracle 原始 outputs 与六专家checkpoint未在当前仓库找到。
用户恢复本Task后，GPU0已启动baseline-rebuild-20261007-r4；2epoch原生GPU smoke通过，
64env/200epoch随机初始化s1母策略已完成，GPU1/2继续四源PointWorld混合拟合。
[母策略资格检查](tools/run/qualify_parent.py)固定64条frame0完整episode，核对45帧保持/之后不掉落，
此次0/64达到稳定抬升，先同输入回放参考动作以区分优化问题与参考/控制/几何问题；
通过后才扩展六专家与连续采集。当前44项CPU工程测试覆盖采集、标签、progress独立抽样、
三臂训练合同、独立评价和资格指标，不代表真实oracle headroom。
后续协议见[Oracle Probe](docs/experiments/probes/P-20261007-consequence-oracle-headroom.md)。

后续诊断：r4首步物体跳回创建原点，GPU reset违反setter/refresh时序。
task-local修复已接入三个native入口；同环境真实8env测试首步位移1.48m→1.43mm，
初始手FK和重复子集reset检查通过，51项工程测试通过。r4不作为有效训练负证据。
旧新均使用graspenv解释器/Dexplore_Inspire任务；已找回原s1 corrected tensor，
其q与接触合同区别于r4canonical。原输入真实reset/FK/三次子集检查已通过，缺失CmLite权重仍需重建，
不将临时随机Cm-off路线称为原母策略复现。
新一轮使用原s1tensor/原single-motion config和修复reset，GPU0完成200epoch Cm-off Probe，
独立64条frame0完整episode中50条通过45帧保持/无后续drop，满足预设8/64数据准备门槛。
该母策略权重仅属于新Probe，不继承旧Validation；缺失CmLite奖励尚未启用。
s3 corrected input已恢复并通过几何审计；下一步仅迁移20epoch并独立资格检查，
其他专家和真实evaluator拟合尚未完成。迁移和评估均核对权重祖先来自owned随机自训练。

旧ref1的reactive future A合同已被ref2替换，旧v1数据禁止进入新训练；未来action trace不再作为输入。
正向才值得进行EWM与后续在线规划；负向先查数据/监督/拟合，不能直接判世界模型无用。

## 2026-10-07 ref2/ref3 当前进展

新s1 parent固定资格50/64；s3从220有界续训至260后固定资格36/64（56.25%），
达到8/64运营准备门槛。其余十条corrected reference完成恢复/几何审计；
六专家中只有airplane_base完成，余五个专家、真实连续数据和三臂fit均未完成。
GPU0目前为他人进程，用户暂不能协调；先完成合同修订，不抢占其他任务。
74项Task工程测试通过，包括请求计划/实际动作分离、三臂屏蔽、interaction坐标、
几何接触约束和当前状态配对。GPU1的8env×128步真实Isaac geometry smoke通过（24.24s），712个forceproxy帧及568个抬升proxy帧均有≤1cm采样距离；
仍缺其他物体/桌面反例和完整采集，不能视为精确contact GT。
四源PointWorld已保存停止step14250；用户要求三源主监督（Oak/GRAB/ARCTIC），
ContactPose仅列为刚性运输辅助，EPIC保持candidate_only。三源已于23:20启动，GPU1/2每卡64，以latest14250权重初始化，
fresh优化器/调度，50000新更新、2026-10-08 10:00截止，另见混合预训练卡。

## 最新执行状态与 ref2 完成边界

2026-10-07晚：GPU0恢复可用，duck260→280短迁移完成，固定资格0/64；
5条曾保持45帧后出现drop。原生duck几何smoke1024帧/533forceproxy帧
均有<=1cm采样gap，reset/FK/子集检查通过；尚未获得桌面假阳性或
完整几何标签验证。六角色中airplane_base已达36/64，duck未过门槛，
其余四角色待训练，真实evaluator数据/三臂fit仍未开始。

修复collector读取native motion目录软链接和遗漏source依赖冻结，
按实际motion跨wave轮转clean及五阶段，记录assigned/triggered覆盖；
不能再用全局env序号分阶段并假设每对象都有clean。单元和fake-driver
测试只验证工程合同，不代替真实collect和标签审计。
三源PointWorld同panel验证会波动，完整状态以混合预训练卡为准。

### 重建后生成新的固定路由

六个实际角色训练并完成各自qualification后，用
`tools/run/prepare_expert_route.py --run airplane_base=<training-run> ...
--run cup=<training-run> --output outputs/consequence-evaluator/<run>/route.json`。
必须提供airplane_base/mixed12/train5/balanced5/duck/cup全部六角色；
脚本核对owned随机自训练祖先、每角色训练身份、六个不同checkpoint、
完整frame0资格及原始trace哈希，默认airplane必须过8/64门槛。
其余弱角色的实际资格数显式保留，不能因路由就绪宣称六专家都可靠。
固定object route沿用既有映射，不从新的测试结果重选。
collector再次核对/冻结这些路由证据，真实progress和preference仍须
独立label检查；当前 route 实例的 `training_allowed=false`，而路由工具会按六个角色的
资格结果动态设置该字段。此工具不训练模型，
不会用一个actor替代六个角色，也不恢复旧Validation身份。

2026-10-08最新：duck260→340的80epoch迁移完成489.07s，固定64条
frame0资格8/64，达到运营门槛但仍较弱。GPU0已接cup260→340，
2epoch工程检查通过，64env正式迁移和端点资格继续；mixed12/train5/
balanced5三角色仍未训练。全Task89项工程测试通过，真实evaluator
采集/三臂训练未开始，不把路由工具就绪当作数据或科学结论。
