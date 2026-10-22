# 结果解读表

| 结果 | 实测数字与含义 | 决定 |
|---|---|---|
| event 全部 ≤t 可得，且实际入口一致 | 不成立。207,579 TRAIN 窗口中有效事件 207,561；type、delta、validity 均依赖未来 teacher | goal 之外还有第二处未统一的条件合同；不放行冒烟 |
| 某字段来自未来 GT / 示教终点 | 真实 `(0,0,36)` 为 CONTACT/11/valid；保持 ≤t 不变、只改未来后变 NONE/0/valid，截断到 t 则 unknown | 报最小合同修复清单；不改 config status，不训练 |
| 不可因果恢复的 runtime 字段，删除会改行为 | 当前 curated 未绑定先验 TaskSpec、scheduler/observer reset，以及与部署一致的力/反馈来源 | 需补录制或绑定独立事前指令；禁止用未来 GT 填补 |
| 探针与真实入口不等价 | 未发生。实际训练 reader options、Steps、DataLoader、collate 共 35 个张量字段逐位一致；32 条中 valid=26、dropout=6 | 入口证据有效；真实事件字段确实保留了 teacher 语义 |
| 聚类 CI 跨 0，帧级不跨 | 未发生，8/8 子集两种 CI 均不跨 0；124 母序列、每子集 2,000 次 bootstrap | 保留旧 TRAIN 面板 B1>B0a 结论；主 CI 改为母序列聚类，不升级为部署优势 |
| 采样索引未增加独立母序列/物体 | 两侧均 1,064 母序列、61 物体；每批母序列 55.68→288，但 source1 比例 11.17%→97.75% | **数据组成变更**，只留机主选择项，不接入、不整批重排 |

今晚 **0 optimizer updates、0 新 checkpoint、0 物理采集**。训练 config、
`curated_sampling.py`、执行器代码、PROJECT_MEMORY 均未修改。

## 范围、证据与决定

Task=`ptv3-shared-wam-v6`，run_id=`tau-contract-20261023-r1`，branch=`main`。
实现 checkpoint 为 `9eeb3fac2ef43f0d9ca7a2c375677946a8c4b5fd`。
工具在本轮开发期间执行；共享 HEAD 随另一会话前移，原产物记录了当时 HEAD。
该 checkpoint 固定最终实现，不声称执行时工作区完全干净。

这次需要区分三件事：事件是否可由因果信息恢复，旧 CI 是否把相关帧当成独立证据，
以及新索引是否增加独立覆盖。最便宜的判别分别是实际 sidecar 全索引统计加未来尾段
反例、复用原面板并补齐未保存的逐帧记录、以及仅索引层模拟。没有新模型或新基线。
CI 属于既有 `P-20261023-tau-fixed-panel-metric` 的统计口径修正。

产物均位于 `outputs/ptv3-shared-wam-v6/tau-contract-20261023-r1/`：

- `event-source-audit.json`：完整 TRAIN/DEV、curated 与 anchor 事件计数，真实/合成反例。
- `event-model-inputs-gt.json`：使用真实 GT geometry reader option 的入口证据，35 个字段。
  初版 `event-model-inputs.json` 使用默认 reader option、比较 34 字段，保留为工程过程；
  正式结论采用前者，二者事件张量相同。
- `paired-frame-records.npz/json`：旧 B1/B0a 的配对误差、mask、母序列/slot/tick/phase。
- `cluster-bootstrap.json`：帧级对照与主聚类 CI，完整 8 子集。
- `sampling-index-design.json`：数据组成选择项的等预算索引模拟。
- `audit-summary.json`：门槛、环境和完整性断言摘要。

## 1. 事件来源：逐字段审计

入口配方是 `local-expanded-recipe-20261021-r1/manifest.json`，其 `event_manifest`
指向同目录 `events.json`；后者实际保存 proximity-event annotation manifest。
本轮沿该指针读取实际 sidecar，不把 recipe 名称中的“plan”当作因果性证明。

生成链是 `ptv3-shared-wam-v5/tools/run/prepare_events.py:39–45`：读取完整录制的
`contact.npy / valid.npy`，调用 `proximity_counts` 和 `annotate_events`。它们是基于
几何 proximity 的 teacher，不是已认证的力接触或计划意图。

| 字段 | 来源判定 | 文件:行证据 | 最小读数 / 反证 |
|---|---|---|---|
| `event_type` CONTACT / RELEASE | **未来 GT** | `ptv3-shared-wam-v5/src/ptv3_shared_wam_v5/event_labels.py:98–103,115–122`，持续 3 帧才确认，再回填 onset；逆序找下一事件 | 真实 t=36 为 CONTACT；只改 >36 的 proximity 尾段后为 NONE；合成例同历史可得到 CONTACT 或 RELEASE |
| `event_delta_steps` | **未来 GT** | 同文件 `:122`，`next_t-t`；不是 scheduler 估计 | 真实值 11；尾段修改后 0；合成 CONTACT / RELEASE 均可为 7 |
| `event_valid`，包括 known NONE | **未来 GT / 录制边界** | 同文件 `:116–124`，未来未知/gap 重置，或要求 `t+24+3-1 < end` | 同一 t=5 clear 历史，长度32为 known NONE，长度31为 unknown |
| `state`，非直接模型 event 输入 | **可含未来** | 同文件 `:99–103`，把确认后的状态回填到 onset | 合成 t=12：完整录制 state=1，只读到12则 state=0 |
| source10 的 frame30 奇数时刻事件 | **未来 GT** | 同文件 `:68–75`，读取右侧真实观测的事件及 validity，再调整剩余帧数 | 直接代码依赖 t+1；不能把 midpoint 条件称作 ≤t 测量 |
| `event_hz=30`、`horizon=24` | **静态合同，≤t 可得** | `ptv3_shared_wam_v6/data.py:68`；`events.py:21,32` | 实际批次 hz 全为30 |
| `delta>24` 窗口外标记 | **继承未来 GT**，24本身是静态边界 | `ptv3_shared_wam_v6/events.py:40–43` | TRAIN 中 130,740 条有效事件在 K24 之外；没有被改成 NONE |
| one-hot 与 `min(delta,240)/24` | **继承未来 GT** | 同文件 `:38–43` | delta=114 产生特征 `[0,0,1,4.75,1]` |
| `event_condition_dropout` 决策 | **≤t 可得**，独立 RNG | `ptv3_shared_wam_v6/data.py:72–77`，seed/step/rank/19281 | 32 条实际 draw 中6条 dropped，固定种子可复现；不是人口比例估计 |
| dropout 后 event 有效性与值 | **仍包含未来 teacher**，约85%保留 | 同文件 `:64–69`：`valid=raw_valid and not drop`，之后 type/delta 置0 | 同一 CONTACT/11，drop=False 保留；drop=True 变 unknown，全零特征 |

上表各路径均位于 `src/task/` 下。完整函数与最小反证在
[audit_event_condition_contract.py](../../tools/audit/audit_event_condition_contract.py)。
真实源重算**整份** annotation 与 stored sidecar 逐位一致，然后才做未来尾段扰动；
没有修改缓存、原数据或未来标签文件。

```text
real s1-00000 / sample_id=[0,0,36], identical contact prefix <=36:
  original:             type=1, delta=11, valid=true
  future held current:  type=0, delta=0,  valid=true
  truncated at t:       type=0, delta=0,  valid=false
synthetic same-prefix next event: CONTACT/7 or RELEASE/7
known NONE at t=5: length32 valid=true; length31 valid=false
```

| 索引总体 | 窗口 | raw valid / invalid | valid NONE / CONTACT / RELEASE | valid delta>24 |
|---|---:|---:|---:|---:|
| 当前 TRAIN | 207,579 | 207,561 / 18 | 2,382 / 19,740 / 185,439 | 130,740 |
| 当前 DEV | 3,356 | 3,341 / 15 | 209 / 381 / 2,751 | 2,087 |
| curated r7 | 6,000 | 5,983 / 17 | 986 / 1,458 / 3,539 | 1,620 |
| 固定锚点 | 512 | 494 / 18 | 71 / 145 / 278 | 173 |

这是各索引去重窗口的 sidecar census，不是历史随机采样的曝光次数或训练后果。

## 2. 实际模型入口

[probe_event_model_inputs.py](../../tools/audit/probe_event_model_inputs.py) import 真实
`tools/run/train_pretrain.py`，不调用 main，不构造模型或 optimizer。正式探针使用
该训练 config 的 `geometry_gt_object=True`，以及同一个 `V6Windows`、`Steps` 和
`collate` 函数对象。`DataLoader(num_workers=0)` 与直接逐项读取后 collate 的全部
**35 个 tensor 字段逐位相同**。worker 数量不改变确定性的 `(index, drop)` 合同。

真实链：`train_pretrain.py:207–216` 构造 DataLoader，`:281` 取 batch，`:289`
只搬运到 device，`:301` 直接 `wrapped(batch)`。探针同时检查该段结构及没有
`batch[...]` 写入。`model.py:139–141` 调用 `self.event_goal(batch)`；
`events.py:52–55` 直接调用本次记录的 `event_features(batch)`。
因此证据针对模型实际接收的事件张量，不只是 scheduler 中间结果。

| 张量 | 形状 / dtype | 实测 |
|---|---|---|
| `event_type` | `[32]` / int64 | 0/1/2 编码，逐条值见 JSON |
| `event_delta_steps` | `[32]` / float32 | teacher 倒计时或 dropout 后0 |
| `event_valid` | `[32]` / bool | 26 true，6 false |
| `event_hz` | `[32]` / int64 | 全30 |
| `event_raw_invalid` / `event_condition_dropout` | 各 `[32]` / bool | 0 / 6 |
| 实际 event features | `[32,5]` / float32 | one-hot3、capped delta/24、far bit；invalid 全零 |
| `goal_object_delta` / `object_goal_valid` | `[32,32,3]` float32 / `[32,32]` bool | 1,024/1,024 query valid；沿用旧 reader，未作 goal 合同修复 |

真实 Steps 批次的前8条（step=1、rank=0、seed=170；不是人工挑选）：

| sample_id | type | delta30 | valid | dropout |
|---|---:|---:|---|---|
| [587,0,72] | 2 | 20 | true | false |
| [955,0,58] | 2 | 2 | true | false |
| [536,0,75] | 2 | 19 | true | false |
| [951,0,85] | 2 | 114 | true | false |
| [90,0,136] | 2 | 129 | true | false |
| [79,0,663] | 0 | 0 | false | true |
| [640,0,190] | 2 | 47 | true | false |
| [193,0,139] | 0 | 0 | false | true |

另有8条受控覆盖样例：CONTACT/11、RELEASE/211、raw-invalid、known-NONE，各取
drop=False/True 一对。有效3条、raw-invalid2条、dropout4条。known NONE 的特征是
`[1,0,0,0,0]`，unknown 是 `[0,0,0,0,0]`；没有把二者混成同一掩码语义。

## 3. 部署对照与最小修复清单

`goal_scheduler.OrderedGoalScheduler / update_schedulers` **不产生 E_next 字段**。
真正部署事件来自 `wam-rl-policy-v3/src/wam_rl_policy_v3/task_scheduler.py`：
`:30–34` 转换成 WAM30Hz packet，`:79–95` 计算事件类型和剩余时间。
v6 `tools/run/samples_efficiency_env.py:140,165–167` 和
`tools/run/evaluate_executor.py:76` 把该 packet 送到 WAM。

| 字段 | 训练实际来源 | 部署实际来源 | 判定 |
|---|---|---|---|
| type | 实际未来 proximity transition | 事前 TaskSpec.events、已观测事件进度；恢复抓取时 CONTACT | **未统一** |
| delta30 | 实际未来 onset−t | ACQUIRE 用计划时刻减 elapsed；TRACK 用计划时刻减已测 source cursor；RELEASE/恢复可为0 | **未统一**，不是同一个倒计时 |
| valid / known NONE | 未来 teacher 的观察覆盖和尾部删失 | 显式计划可用性；计划耗尽才 known NONE，无计划则 unknown | **未统一** |
| hz / feature 编码 | 30Hz、K24、cap240 | 同30Hz，经同 `event_features` | 数值格式一致，来源语义未统一 |
| dropout | TRAIN 随机遮蔽15% | 推理不随机 dropout；缺事件显式 unknown | 常规训练正则，但不能消除上述错配 |

`online_runtime.py:160–162` 默认事件 unknown，或原样接收 caller event；
`:251–266` 经 online→collate→模型。runtime 本身没有把 teacher 事件变成估计器，
也没有从未来真实接触取得事件。探针实际调用 TaskScheduler，得到 CONTACT
`30→15 frames`（now `0→0.5s`），恢复 ACQUIRE 为 `0 frames`。

TaskSpec 的来源也必须单独核实：`task_annotations.py:75–88` 的
`teacher_rollout_task_spec` 用完整旧 teacher rollout 的力序列生成事件，
`:40–64` 也有 reference contact 的离线事件生成。这种**旧轨迹预先声明的计划**可以
作为新执行的已知输入，但不是新执行未来 GT；`source_role` 和 available-before-reset
必须绑定。不能因为部署代码叫“planned”就把同次执行的未来 teacher 合法化。

最小合同修复清单（仅提出，不实施）：

1. 绑定部署前已知的 TaskSpec：event kind/source_frame、source_fps、object_path、
   source_frames、terminal_mode、source_role 与意图来源；不要求预存每 tick E_next。
2. 录制同一部署 TaskScheduler/FeedbackObserver 的 reset/snapshot，以及它所消费的
   ≤t pose/hand/velocity、contact_forces/force_valid、support/workspace 信号或直接
   因果反馈包。缺哪个就补哪个；human proximity 标签不能冒充机器人力反馈。
3. 用同一状态机重放，再在 collate 后与 online 模型入口比较 type/delta/valid/hz，
   包括 known NONE、unknown、recovery、release、far event 与 dropout 后实际 tensor。
4. 当前缺这条记录链的部分只标“需补录制/绑定”，不拿 GT 填 runtime 字段。
   删除事件也不是无影响修复：scheduler 的 event_index、release 授权和 mandatory
   path milestones 都依赖计划事件，移除会改变行为。

另一个待机主决定的合同选项是训练/部署统一 unknown event mask；这会改变训练条件，
不能伪装成保留原配方的合同修复。本轮未实施，也未修改 config status 或启动任何训练。

## 4. 母序列聚类 bootstrap 取代帧级主 CI

旧 receipt 仅有聚合统计，无法据此恢复母序列内协方差。为补缺失记录，在 GPU7
对同一个 step011000、同一512行/顺序、相同 reader、相同随机种子与 mask 做一次
**只读误差导出**；仅 B1 与既有 B0a，未新增/重跑其它基线。耗时18.31s，GPU峰值
分配679,267,328 bytes；无梯度、optimizer、训练或 checkpoint 写入。
B1/B0a overall mean 重现为23.409973/143.425247mm；8个子集帧级原CI按原seed全部
重现至1e-5mm。旧 baseline JSON 未覆盖。

主抽样单位为124个 **sequence_id 母序列**，同母序列的多个 object slot、窗口、
全部有效未来帧一起带入；每次有放回抽124个母序列，重复母序列按抽中次数计权。
每子集2,000次，percentile95% CI。每个子集都保留整个124母序列总体，未含该相位
的母序列在该子集贡献0帧。统计量保持原来的**全部有效帧配对差中位数**，不是
母序列中位数再取中位数，也不是 mean EPE 的 CI。

| 子集 | 有效帧 | 有值母序列 | 原帧级95% CI mm | 主聚类95% CI mm |
|---|---:|---:|---|---|
| overall | 12,216 | 124 | [-43.878,-40.660] | **[-50.817,-35.526]** |
| first8 | 4,091 | 124 | [-18.338,-15.854] | **[-21.198,-13.684]** |
| contact-window | 7,416 | 116 | [-32.658,-29.598] | **[-36.580,-26.002]** |
| approach | 2,520 | 79 | [-232.056,-193.485] | [-301.609,-106.713] |
| close | 2,520 | 100 | [-53.081,-50.050] | [-57.010,-47.247] |
| contact | 2,520 | 104 | [-22.787,-19.428] | [-27.427,-15.443] |
| hold | 2,520 | 102 | [-34.099,-26.425] | [-43.537,-20.113] |
| unknown | 2,136 | 17 | [-26.238,-22.170] | [-103.138,-16.978] |

8/8子集的原结论均未反转，但帧级 CI 明显低估这里的组内相关性。
unknown 仅17个有值母序列，区间尤其宽。B1 的 approach mean EPE 仍为29.897742mm，
本轮没有改善它。该结果仍只描述**旧条件下固定 TRAIN 面板**，event future leak
阻断了把它直接解释成部署可得的 τ 轴优势。

[panel_uncertainty.py](../../src/ptv3_shared_wam_v6/panel_uncertainty.py) 实现整母序列重采样；
原 [scorer](../../tools/run/score_tau_panel_trivial_baselines.py) 以后默认把聚类结果写入
主 `paired_difference_median_bootstrap`，帧级对照放入
`paired_difference_median_frame_bootstrap`，输出 schema升级v2并保存逐帧记录，
避免再次只有汇总而无法更换 CI。未改变误差、mask、窗口和原总体中位数定义。

验证：2项新增测试分别对照显式整母序列拼接重采样，并构造“重复帧使帧级显著、
但母序列级跨0”的负控制；连同已有事件边界测试，**3 passed、5 deselected，2.07s**。

## 5. 低成本索引：只作为数据组成选择项

比较对象是**同一6,000行**上的现有、尚未接训 `CuratedTauSampler` 与候选索引。
不是把历史 B1 的 V6Windows 随机采样冒充 curated 采样。
参考拓扑取历史曝光账本的 world3/local96/global288；模拟256批共73,728 draws，
seed170。这只是归一化索引统计，**不是 optimizer 预算或启动授权**。

设计：共享一个母序列循环排列及 cursor，先选288个不同母序列，再切给各 rank；
按全局阶段 deficit 选择该母序列已有阶段，各母序列/阶段内按 tick/slot 排序、循环
推进时间 cursor。四主要阶段目标各24.75%，unknown1%；没有只强化 contact。
`int32 window_ids` 仅24,000 bytes，另有 offsets 与少量 cursor；不需要读几何、
重算 contact 或重建数据缓存。没有物化训练顺序、修改生产 sampler 或接线。

| 指标 | 现有 curated sampler | 候选索引 |
|---|---:|---:|
| 每global batch母序列数期望估计 | 55.684 | 288 |
| 同批重复母序列占比 | 80.665% | 0% |
| 同批完全重复窗口占比 | 43.920% | 0% |
| 同批时间近重复窗口占比 | 75.959% | 0% |
| 每原始窗口平均暴露 | 12.288 | 12.288 |
| 曝光p50 / p95 / max | 1 / 26 / 2,738 | 12 / 27 / 69 |
| 尚未曝光窗口 | 1,597 | 177 |
| 访问过母序列 / 物体 | 1,064 / 61 | 1,064 / 61 |
| 母序列/物体池新增 | 0 / 0 | 0 / 0 |

近重复定义：同批同sequence/slot另一个窗口的 `abs(tick difference)<24`，重复
tick也算；报告的是落入任一近邻关系的 **draw占比**。它是256批Monte Carlo
期望估计，不能与数据池结构性61.233%的去重窗口比例混用；后者两侧保持不变。

| 阶段 | 原6k池窗口占比（两侧不变） | 现有抽样draw占比 | 候选draw占比 |
|---|---:|---:|---:|
| approach | 20% | 15.853% | 24.750% |
| close | 20% | 23.732% | 24.750% |
| contact | 30% | 20.059% | 24.750% |
| hold | 29% | 29.305% | 24.750% |
| unknown | 1% | 11.051% | 0.998% |

代价是放弃 source 均匀权重：source1 draw 占比11.1735%→97.7458%，其他每个
source约0.28%。这是“母序列均衡”在当前极不均匀来源组成下的直接后果。
固定母序列循环也有周期性，不代表不断获得新独立数据；时间轮换仅作用于已有行。

**决定：不整批重排，不接入训练，只留机主选择项。** 独立母序列/物体数没有增加。
这是数据组成/抽样比例变更；不能将潜在结果与条件修正、预算缩减或新数据覆盖混合归因。

## 复现命令与写盘字节数

环境：`dexplore_repro_py38_torch222_cu121`，Python3.8.20、NumPy1.24.4、
PyTorch2.4.1+cu121。除一次缺失逐帧记录导出外均为CPU读盘/统计；没有训练。
输出文件禁止覆盖，下面用 r1 下不存在的 `reproduce/` 保留原证据。

```bash
tau_python=/home2/wyy/miniconda3/envs/dexplore_repro_py38_torch222_cu121/bin/python
tau_tools=src/task/ptv3-shared-wam-v6/tools/audit
tau_out=outputs/ptv3-shared-wam-v6/tau-contract-20261023-r1/reproduce
export PYTHONDONTWRITEBYTECODE=1

"$tau_python" "$tau_tools/audit_event_condition_contract.py" \
  --recipe outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json \
  --curated-manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --anchor outputs/ptv3-shared-wam-v6/tau-curated-20261023-r2/train-anchor.json \
  --output "$tau_out/event-source-audit.json"
"$tau_python" "$tau_tools/probe_event_model_inputs.py" \
  --recipe outputs/ptv3-shared-wam-v6/local-expanded-recipe-20261021-r1/manifest.json \
  --training-config outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/config.json \
  --source-audit "$tau_out/event-source-audit.json" --output "$tau_out/event-model-inputs-gt.json"

# 已持久化的 r1/paired-frame-records.npz 足以直接重算 CI，无需再次模型推理。
"$tau_python" "$tau_tools/summarize_tau_cluster_ci.py" \
  --records outputs/ptv3-shared-wam-v6/tau-contract-20261023-r1/paired-frame-records.npz \
  --reference outputs/ptv3-shared-wam-v6/tau-fixed-panel-b0-20261023-r2/baseline-b0.json \
  --output "$tau_out/cluster-bootstrap.json"
"$tau_python" "$tau_tools/design_sequence_sampling_index.py" \
  --manifest outputs/ptv3-shared-wam-v6/tau-curated-20261023-r7/curated-manifest.json \
  --output "$tau_out/sampling-index-design.json"

# 仅在逐帧产物丢失时补导出；运行前重新检查 GPU7 资源，无训练。
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
"$tau_python" "$tau_tools/export_tau_panel_frames.py" \
  --checkpoint outputs/ptv3-shared-wam-v6/flowmatching-overnight-20261022-r2/training/step011000.pt \
  --reference outputs/ptv3-shared-wam-v6/tau-fixed-panel-b0-20261023-r2/baseline-b0.json \
  --output "$tau_out/paired-frame-records.npz" --gpu 7

PYTHONPATH=src/task/ptv3-shared-wam-v6/src:src/task/ptv3-shared-wam-v5/src \
"$tau_python" -m pytest -q -p no:cacheprovider \
  src/task/ptv3-shared-wam-v6/tests/test_panel_uncertainty.py \
  src/task/ptv3-shared-wam-v6/tests/test_v6_training.py \
  -k 'event or cluster or repeated_frames'
```

写盘前 `/home2` 可用148,422,340,608 bytes。r1共8个产物，逻辑大小合计
**163,972 bytes**，实际分配 **180,224 bytes**；checkpoint新增 **0 bytes**。
这些数字只计该输出目录的文件，不包括Git中的代码/Markdown。
