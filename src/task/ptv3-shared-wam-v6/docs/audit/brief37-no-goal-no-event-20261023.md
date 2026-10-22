| 结果解读表（实测） | 含义 | 决定 |
|---|---|---|
| 最小检查①/②/③ **3/3 PASS**；9 source代表与生产Steps入口 goal逐位一致；**6,000/6,000**窗口监督保留；RELEASE **3/3**检查通过 | 路径A的 no-goal / no-event 合同已落地 | 条件门通过，未追认原goal来源合法 |
| 资源/身份 preflight **7/7 PASS**；GPU4/5/6各空闲 **25,430,065,152 B**；磁盘空闲 **146,681,860,096 B** | 授权池按brief34 spare豁免通过，未抢占其他卡 | **05:59:24**启动唯一一次尝试 |
| 启动后 **FAILED_DATA_LOADER，0 updates，0 optimizer样本曝光**；06:02:06已退出 | AF_UNIX路径过长阻断首batch传输，随后120s超时；不是模型/τ负证据 | **未重启**；保存初始化和失败现场，修基础设施后等后续裁决 |
| init@A：前8帧hand **16.1694mm**；固定contact hand **59.2219mm**；前8帧object **11.2523mm**；同条件重复预测差 **0mm** | 同一A条件的真实初始化TRAIN分布内基准已取得；trained@A不存在 | 不能据此宣称τ改善或goal删除效应；旧B1旧事件条件读数不进入配对 |
| 执行后完整任务成功/安全中止 **未测**；本轮新rollout **0** | 任务规范未降级，但没有执行收益读数 | 不声称改善进入执行；不外推τ路线被否证 |
| IPC修复后真实Listener+spawn tensor通过；完整生产reader/Steps/96 batch/4 spawn workers CPU模型入口通过；回归 **28 passed** | 故障已有可复核工程修复，未再次运行训练 | 只交证据；自动重启false仍生效 |
| 训练早于07:15硬停已退出；磁盘stop **0**次、OOM **0**、非有限值 **0**；GPU4/5/6各 **2MiB /0%** | 资源全部释放；无第二次训练 | 保留stop20GiB /warn30GiB /artifact alarm32GiB，不删历史outputs |

## 三项检查证据

机主brief37已裁决路径A，本轮没有再论证A/B/C。原goal来源结论保持 **可认证0/6000、确证未来0/6000、不确定6000/6000**；不重复brief31/32/35普查或r9重放。训练显式为 **no-goal / no-event**；B不采用，C不作为今晚前置。

### ① 真实模型入口一致

生产reader与runtime调用同一个 `src/ptv3_shared_wam_v6/goal_policy.py:15`；不是两个近似实现。`data.py:47–64`含spawn/serde传递与条件清零；`online_runtime.py:146–169`在goal转换前强制None并使用同一构造器，`:195–196`涵盖full/forecast-only；`model.py:128–132`也防止direct encode重新暴露旧goal。`tools/run/train_pretrain.py:125–128`对TRAIN/DEV传递策略，DEV保持原索引而非误用6000 TRAIN索引。

[9 source和Steps入口](brief37-no-goal-no-event-20261023-r1/checks.json) / [原始位模式](brief37-no-goal-no-event-20261023-r1/model-boundary.npz)：真实 `SharedWAMV6.forward_pre_hook` 捕获生产V6Windows→collate与真实OnlineV6Runtime.predict→online→collate→model调用；拦截发生在任何网络/geometry forward之前（geometry仅构造seam），没有optimizer或backward。复用brief36已确认的方法与发现，不重复其普查。

| 实际张量 | 9-source reader/collate/model与runtime | 生产Steps8窗口 | 修复后完整96 batch /4 spawn workers |
|---|---|---|---|
| goal_object_delta | torch.float32 `[9,32,3]`，864个+0 | `[8,32,3]`全+0 | `[96,32,3]`全+0 |
| goal_object_points | torch.float32 `[9,32,3]`，864个+0 | `[8,32,3]`全+0 | `[96,32,3]`全+0 |
| object_goal_valid | torch.bool `[9,32]`，288 False/0 True | 256 False/0 True | 3072 False/0 True |
| event_type/delta/valid | canonical int64/float32/bool，全0/False；hz=30为时间元数据 | `[8]`全unknown；已验收事件配方不变 | `[96]`全unknown，known-NONE=False |

三个goal字段训练/runtime逐位相等 **True/True/True**；runtime传有效goal+RELEASE事件与传None的goal张量也逐位一致。对旧goal张量注入全1后经A清理，**整个batch所有字段逐位恢复一致**；online的有效goal vs None负控制也核对所有字段一致。v5 `data.py:35–40`的features只依赖H4历史；`:60–61`原endpoint字段被同一A构造器删除，不改名为waypoint。scene_target和trajectory_target保留为监督标签，不作为A目标信息。

`goal_valid=False`是删除条件，**不是**zero goal+valid=True；该反例测试保留。回答：**是，新A政策下训练与部署goal缺失输入语义已经一致；旧goal来源仍未认证。**

### ② 删除条件不删除监督

全量6000窗口只为监督不变量进行对照，不再扫描原始provenance。旧reader与A reader所有非goal字段 **6000/6000逐位不变**，包括手物轨迹、contact/part标签、GT2048未来物体几何及所有mask；同一有序6000元组及原CuratedTauSampler不变。

| 有效性 /评分分母 | 旧 | A |
|---|---:|---:|
| 有效手监督窗口 | 6000 | 6000 |
| 有效物体监督窗口 | 6000 | 6000 |
| hand有效point-frame | 6,002,577 | 6,002,577 |
| scene有效point-frame | 4,608,000 | 4,608,000 |
| contact有效query-frame | 293,734,400 | 293,734,400 |
| part有效query-frame | 64,125,646 | 64,125,646 |
| 前8帧hand评分分母 | 2,001,216 | 2,001,216 |
| 固定contact窗口hand分母 | 4,329,318 | 4,329,318 |
| GT2048未来标签完整窗口 | 6000 | 6000 |

[真实模型入口监督张量](brief37-no-goal-no-event-20261023-r1/supervision-boundary.json) / [NPZ](brief37-no-goal-no-event-20261023-r1/supervision-boundary.npz)：8窗口实际模型入口，hand/scene/contact mask分母分别 **6048/6144/393216**，目标SHA与旧batch逐位核对，改变字段数0。修复后[96 batch /4 spawn入口](brief37-no-goal-no-event-20261023-r1/spawn96-supervision-boundary.json)同样通过，mask分母 **82047/73728/4517888**，全部监督字段不变；这是CPU数据/入口工程检查，**0次网络forward、0 optimizer**，不是第二次训练。

真实 `objectives.loss_components` 测试确认hand/scene/contact/part/flow_hand/flow_scene的加权分母与固定预测下分子均完全不变；`objectives.py:34–46`和v5 `objectives.py:14–38`不以goal mask计算loss有效性。`geometry_gt_contact.py:38–67`用原interaction/object/hand/trajectory masks和GT2048标签，不读goal_valid；这些输入6000/6000逐位不变。没有删除geometry contact loss或接触预测目标，也未更改任何loss权重。

### ③ 执行与评分仍持有真实任务

修改只在WAM包内部删除条件，不传回或修改caller目标、TaskSpec或SchedulerDecision。只读调用现有 `wam-rl-policy-v3/tests/test_terminal_release_timing.py` 三项测试：声明帧释放、禁止跳过有序transport、释放后终点/support要求均通过。TaskSpec.packet调用前后完全相同，terminal_mode仍RELEASE。`task_scheduler.py:55–63`保留完整spec/path；`:195–213`要求RELEASE阶段真实released+stable+终点/path_complete+support并dwell，未变为仅抬升。`executor_contract.py:12`仍消费真实decision；该Task任何文件均未由本轮修改。

模型入口是unknown goal/event，外层任务信息仍由调度器/执行器/评分器保有。这是接口与纯状态机证据，**不冒充新物理rollout或执行成功率**；执行器线与source-only A0 τ测量仍独立。

## 训练启动 / 停止实测记录

新冻结文件：[tau_no_goal_no_event_v1.json](../../configs/tau_no_goal_no_event_v1.json)；执行config另存 `tau_no_goal_no_event_train_v1.json`。旧 `tau_event_free_v1.json`原样保留，SHA256仍 `7bfcd6e569bb205b88439a9d2bf056eff8421198a03fe614c69d1538d6308044`；原curated config status与goal_missing_policy没有改写。

- 唯一起点HEAD：`b60e61db36ef97f764917c53cef2eb99ffa67a59`；修复代码冻结HEAD：`e37ef933e83ec3a7ad715690829cde33fc272fcd`。后者**未用于另一次训练**。
- 新配方SHA256：`36872da24a5dc72cc5580d94a048b5fd7a6ed111b21e7053a4b17acbeff9fc5d`。
- r7 manifest SHA256：`5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`。
- 有序tuple SHA256：`d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7`；口径 `[source_id,[sequence_id,slot,tick],tick]`、原selection顺序。
- [preflight](brief37-no-goal-no-event-20261023-r1/preflight.json) **7/7 PASS**，实测磁盘空闲146,681,860,096 B；GPU4/5/6各used2,097,152 B、free25,430,065,152 B。明确记录机主4/5/6/7分区、0–3属zyc、7执行器优先，spare仅在授权池有富余时要求；无其他门豁免。
- [launch](brief37-no-goal-no-event-20261023-r1/launch.json)：**05:59:24+08:00**，launcher PID289790、rank PID289879/289880/289881、supervisor PID289782，GPU4/5/6；预计最多剩余4535.7s、写盘16GiB；checkpoint参考文件实测1,217,116,001B/份，按每1000更新约1.217MB/update只是容量投影，**本次没有更新故无实测持续增长率**。
- 保存真实初始化 `tmp/tau-no-goal-no-event-20261023-r1/training/step000000.pt`：**405,716,522B**；626模型state tensors与source step011000逐位相等；optimizer state为空，未恢复旧optimizer。见[初始化](brief37-no-goal-no-event-20261023-r1/initialization.json)。
- [停止](brief37-no-goal-no-event-20261023-r1/stop-normalized.json)：**06:02:06+08:00**，首batch传输前失败；**0 updates、0 optimizer曝光、loss/LR/吞吐/剩余训练时间均无有效读数**，因此200–300 updates一栏不能填造。
- 原因：[原始错误日志](brief37-no-goal-no-event-20261023-r1/raw-train-launch.txt)是 `multiprocessing.resource_sharer.Listener -> AF_UNIX path too long`，随后DataLoader120s超时；不是OOM、磁盘告警或数值发散。独立CPU最小复现tempdir本身116B即超过socket路径上限107B；修复后实际Listener地址60B，真实spawn tensor传输和96 batch/4 workers生产路径通过。修复只限制IPC路径长度，不变模型、优化参数、数据或条件。
- 原训练程序捕获loader_failure后返回0，原supervisor只按returncode写reason=COMPLETED，**这个原字段错误**。保留[raw receipt](brief37-no-goal-no-event-20261023-r1/raw-stop-receipt.json)与FAILED_DATA_LOADER嵌套result，另写normalized receipt纠正；修复后loader失败抛异常，supervisor优先识别result状态，含回归测试。不能把exit0当训练成功。
- watchdog磁盘20GiB stop、30GiB warn、产物32GiB alarm均保留；[原watchdog日志](brief37-no-goal-no-event-20261023-r1/raw-watchdog.jsonl)。07:14:30预留保存、07:15对**本次新session进程组**硬停；实际06:02已退出，无需等待07:15。未调用kill、未碰2730517–2730520或任何他人PID。资源核验GPU4/5/6已各2MiB/0%；其他卡任务不动。[退出后实测](brief37-no-goal-no-event-20261023-r1/resources-after-stop.json)。

**本轮没有第二次启动**，automatic_restart=false与唯一一次尝试现场保留。CPU工程修复已就绪，后续是否重新准许运行须单独裁决，不能把启动失败记录抹掉重试。

## 固定面板三行读数

| 同一A条件，固定TRAIN面板 | 前8帧hand EPE mm，cluster95%CI | 固定contact-window hand EPE | 前8帧object EPE | 两次预测变化量 |
|---|---|---|---|---|
| init@A=真实初始化@A | **16.1694 [15.3260,17.0066]** | **59.2219 [53.8293,64.7564]** | **11.2523 [10.1101,12.3994]** | 同条件重复bit-exact；hand/object重复相等，hand max_abs=0mm |
| trained@A | **N/A：0 updates** | N/A | N/A | 无更新后权重，配对变化/CI不可测；不能把init重复冒充训练改善 |
| 额外“真实初始化权重@A” | **不另设**：实际初始化就是step011000 | 同init@A | 同init@A | 不为凑三格改初始化 |

[面板结果](brief37-no-goal-no-event-20261023-r1/panel-init-A.json)。512固定锚点、124母序列，聚类bootstrap **4000次**；不是相关帧独立CI。原contact window定义沿用“未来标签中any有效surface contact>0.25”，未按预测事件选帧；未来标签只评分。mask与评分分母不变。27.483s、GPU7空闲份额，起前free20,297,285,632B、峰值allocated641,889,792B，未占训练卡。

原始配对预测/误差和count保存在 `tmp/tau-no-goal-no-event-20261023-r1/panel-init-A/paired-panel.npz`，供后续同合同配对；**不携带旧事件条件的历史B1入对照**。这里只有TRAIN分布内初始化读数，没有DEV泛化或执行收益结论。

## 预注册阈值对照（预注册，非实测）

相对真实初始化@A：前8帧hand和固定contact hand EPE均降≥5%、前8帧object EPE恶化≤5% → τ预测改善。对应当前init的数字门槛为hand≤15.3609mm、contact≤56.2608mm、object≤11.8149mm；这些是**预注册阈值计算，不是训练实测**。

20个配对reset完整任务成功净增≥4、安全中止不增加 → 改善进入执行。本轮不新增rollout，执行门未测。只离线改善写“τ预测改善，执行收益未达门槛”；两层无改善写“本轮训练未达改进目标”，均不外推“τ路线被否证”。本次是**基础设施在首更新前失败，阈值不可评价**，不能充当任何一种模型效果负结论。

## 复现命令

环境：`/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python`，Python3.8 / torch2.4.1+cu121；正式训练严格FP32，world3/local96/global288、seed170、原模型与原loss权重不变。

```bash
# 28 passed in4.62s；CPU工程检查，没有训练更新
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python -m pytest -q -p no:cacheprovider \
 --basetemp=src/task/ptv3-shared-wam-v6/tmp/brief37/tests-reproduce \
 src/task/ptv3-shared-wam-v6/tests/test_loader_ipc.py \
 src/task/ptv3-shared-wam-v6/tests/test_no_goal_no_event.py \
 src/task/ptv3-shared-wam-v6/tests/test_v6_goal_contract.py \
 src/task/ptv3-shared-wam-v6/tests/test_event_free_preflight.py

# 已完成的6000监督不变量审计复现；必须新output；不需要再跑provenance普查
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/check_no_goal_no_event.py \
 --output src/task/ptv3-shared-wam-v6/tmp/brief37/checks-reproduce

# IPC修复后的完整生产reader/Steps/96 batch/4 spawn workers，入口即停；0optimizer
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/capture_A_supervision_boundary.py \
 --batch-size 96 --workers 4 --output src/task/ptv3-shared-wam-v6/tmp/brief37/spawn96-reproduce

# init@A inference-only，固定512 TRAIN/原contact窗口/同一unknown事件；需要GPU7空闲份额
/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python \
 src/task/ptv3-shared-wam-v6/tools/run/evaluate_no_goal_no_event_panel.py \
 --checkpoint src/task/ptv3-shared-wam-v6/tmp/tau-no-goal-no-event-20261023-r1/training/step000000.pt \
 --output src/task/ptv3-shared-wam-v6/tmp/brief37/panel-init-reproduce --gpu 7 --batch-size 4

# 原始唯一启动的完整命令及环境见launch.json；此处仅记录，不再执行。
# ONE_SHOT.lock保留，禁止抹掉失败现场或自动重启。
```

## 写盘字节数

预估16GiB，实际完整运行目录字节见下方冻结账本；初始化405,716,522B，init面板包含原始预测/误差，未保存任何更新后checkpoint。本轮产物远低于32GiB，未删任何历史outputs。账本统计代码/审计文档/任务内本轮runtime与工程fixture，排除Git object、他线写入和既有数据。

完整运行目录 **417,020,827 B**（分配417,071,104 B）；工程fixture **5,905,844 B**；审计证据（不含账本自身） **6,024,643 B**。代码/文档和各目录分项见[byte-ledger.json](brief37-no-goal-no-event-20261023-r1/byte-ledger.json)。账本记录的是冻结瞬间的完整文件大小，不将旧数据或他线增长归本轮。
