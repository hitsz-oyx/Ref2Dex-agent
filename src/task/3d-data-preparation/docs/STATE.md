## 2026-10-18 dense overlay 内容绑定已完成

按用户28361da审计，新增manifest v2：逐NPZ SHA-256、recipe中的manifest摘要、checkpoint
中的数据身份。实际加载验证同一份字节；修改480个非query点或替换manifest都会被拒绝。
`tools/run/freeze_v4_surface_recipe.py`已在远端生成独立
`data/recipe-auditbound-20261018-r1`，复用全部1,078个原NPZ，未重新采样、未覆盖旧版本。
修正版WAM-v4代码7b6d28a5已完成真实四卡训练与恢复smoke，详见WAM-v4实验卡。

## 2026-10-18 WAM-v4 真实 distinct512 几何派生产物已验收

九源 canonical 几何审计发现 main3 的 252 个对象已有 512 个独立点，native 目标几何
有 1,078 个不足（UniDex 1,074、TACO 2、HOT3D 1、DexYCB 1）。已从原始 CAD 三角面
补采样到真实 512 点和单位法向，保留每个 native 对象原有 32 个 flow query；不再循环补齐。
原 v3 payload、geometry、checkpoint 均保留。局部数据合同为
`ref2dex.wam-v4-target-object.v3`，不是模型或实验版本号。

远端派生入口为
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v4-dense-20261018-r1/data/recipe-dense512-20261018-r1/`。
2,780 个审计几何均满足 distinct512/unit-normal 合同；对补密后受影响的原已筛窗口复查
TRAIN 635/DEV 242 个，恢复 1 个 TRAIN 窗口，原已保留集合不缩减。实际 TRAIN 九源和
DEV 八源 mixed reader/collate 均通过；当前冻结 native 快照没有 source8 DEV，不伪造该面板。
source probabilities 和 v3 TRAIN normalization 数值保持继承，未重新拟合统计。

采样工具 `tools/run/prepare_v4_dense_surfaces.py`、远端配方工具
`tools/run/prepare_v4_dense_recipe.py`、独立几何审计
`tools/audit/audit_v4_surface_geometry.py`。本地 CAD 处理 52.92s/约12MiB，远端配方处理
27.06s；均为纯几何 CPU 作业。Python 3.8.20/graspenv；代码 `992962ec`，相关测试本地和
远端均 18 passed。真实数据回执在
`outputs/ptv3-shared-wam-v4/dense512-remote-evidence-20261018-r1/data-result.json`，
WAM-v4 实验卡记录四卡接口与后续训练。几何验收不构成机器人收益结论。

## 2026-10-15 DexYCB全10被试补齐已验收

用户已授权补齐。原始1000段/72,746帧全量canonical已落盘，官方s1 700 TRAIN/100 DEV/200 TEST；清洗后699 TRAIN/100 DEV，严格准入3,096 TRAIN/382 DEV窗，TEST独立不参与拟合。subject10全部7,282标签文件与既有官方原始数组一致；复用CAD，无图像/MANO重新估计，无相机重复扩样。九源扩充配方本地和远端均完成，统计数值一致、其余源准入集合保持。当前80k冻结运行未改输入。详见[扩产记录](research/2026-10-15-dexycb-full-expansion.md)及v3当前状态。

# 当前完成情况

## 2026-10-15 native→完整v2 GPU工程接入完成

独立代码快照`ba50a60a`在GPU3通过98,591,142参数完整v2模型的四步/batch12检查（15.52秒）；
混合coverage含source4/5/6/8/9/10，hand/scene两项loss均传至PTv3/shared/hand encoder/goal；
invalid target扰动不改变loss，geometry/shared参数组均实际更新。GPU物理显存峰值3,053MiB，41项Task测试通过。

`native-training-recipe-20261015-r1`绑定冻结TRAIN索引、每源充分统计及抽样草案：source4/5/6/8/9/10
约40.19/13.66/20.83/1.00/5.90/18.42%，来源内均匀window；common标准化按同一概率合成，std floor1mm。
保留现有global masked-point hand+scene loss，没有source loss multiplier；有效插值标签参与、标记另存。
此为新增六源工程配方，未把原main3并入，尚未建立长训收益或训练准入。

原模型接口修改曾触发既有四卡run源码保护，接口已恢复，用户已接手训练恢复；后续全部在独立快照实施，
仅同步本Task的新接入文件，未继续改动原v2模型/训练源码。r1因瞬时六卡上限保护0更新退出，保留回执，r2重跑通过。
详见[接入结果](research/2026-10-15-native-v2-integration.md)。

## 2026-10-15 用户授权清洗、TRAIN统计和encoder兼容版本均已完成

- 决策：用户明确要求开始数据处理并持续推进；在现有main上新增版本化清洗流程，不新建分支，不修改原始NPZ/CAD、既有checkpoint或其他进程。
- 证据：原论文和原生帧号支持source4/5/6/8/9的30Hz输出合同；HO-Cap仍nominal，HOI4D有1,251段与15fps/20秒一致、75段冲突或缺证据。H2O独立TEST的220个文件有明确归属，不是损坏文件。
- 行动：冻结逐行source lineage，单位化非零法线；仅移除不属于固定query的零法线surface点，query受影响则隔离对象；按QA、clock gap、完整target horizon与至少一侧手的future覆盖率过滤窗口。单手样本不因另一手缺失而拒绝。
- 成本与停止条件：纯文件/数组CPU处理，4workers、首次30分钟、单scope2GB，剩余空间低于20GiB或STOP即终止；完成后做独立端点复算和静态可视化。24项数据合同测试通过，独立审查发现的完成标记发布顺序与索引漂移保护已修复。
- 边界：保持`training_allowed=false`，清洗快照不自动等于模型训练准入；先处理六源，再独立转换HOI4D的一致子集。没有新的外部授权需求。

结果：`native3d-clean-mixed-20261015-r1`已完成，37,591段/2,638几何；verified TRAIN427,707、DEV105,952，
HO-Cap另8,933个nominal DEV只留all索引。隔离TACO/187涉及9段、HOT3D无合格窗口2段，HOI4D时钟冲突/缺失75段；
HOI4D另1,251段已保留300原观测并显式插入299半步。全2,638几何/query检查与96窗独立端点复算通过，
最大误差2.81e-8m；七源正/反混合batch的provenance与插值标记保留，最终35项Task测试通过。

TRAIN统计`native3d-train-moments-20261015-r2`使用`38a31e31`，CPU4workers、30分钟/100MB，512.41秒完成；
只读verified TRAIN引用的2,096份CAD与30,029个TRAIN payload，不读取DEV/TEST；按源保存24×XYZ hand/scene的count/mean/M2，
同时生成当前观测workspace不兼容清单，不删窗。r1独立审查发现额外读取了542份非TRAIN使用CAD，
这些几何未进入数值统计；已通过STOP安全边界终止并保留记录，修复读取选择与测试后重跑。
详见[清洗结果与统计记录](research/2026-10-15-native-cleaning-results.md)。

最终新增`native3d-clean-encoder-20261015-r1`（`bf5f9a89`，90.53秒）：只按当前观测的固定体素域排除UniDex
41 TRAIN/6 DEV窗，保留427,666 TRAIN/105,946 verified DEV；37,591段都有剩余窗。对应充分统计已扣除41个TRAIN窗口贡献，
其余来源统计不变；DEV未进入拟合。完整版本/原始数据保留，所有flag仍false。
全542,592个parent窗口scene点数≤3,072，不触发默认4,096 cap；最终view几何、96窗、七源混合接口再次通过。

## 2026-10-15 04:47 native→Shared-WAM v2 字段合同只读审计

- 决策：继续不改 production loader、不翻转任何 `training_allowed`、不启动混合训练；先用最便宜的真实窗口逐字段审计确认 adapter 边界。
- 证据：`tmp/goal-supervision-20261015/native-v2-field-contract.{json,md,png}` 对 source4/5/6/9 各取一个 engineering window，现有 `forecasting_collate` 原样运行；四源模型必需字段均存在且 batch 数值有限，`point_valid`、`object_valid`、`goal_object_delta` 的派生/重算值分别与 xyz 长度、对象 mask、goal−start 一致。
- blocker：`ptv3_shared_wam.egodex_data.forecasting_collate` 对四源一致丢弃 `source_profile`、`anchor_world_to_sample`、`training_allowed`；source4/source5 的一个混合窗口对保留 source/sample 顺序通过，但两源的 local `sample_id` 相同，说明 source ID/lineage 不能省略，且 provenance 丢失仍未解决。因而必须保留 source-profiled canonical record → admission manifest → dedicated native→v2 adapter 的 fail-closed 设计。该审计是接口/工程证据，不是 reader 全覆盖、时钟、几何质量或训练效果结论；没有改变准入门禁。

- 04:57 CST 监督快照后的 5 秒 live-manifest semantic drift probe 仍为 `STABLE_SEMANTIC_CONTENT`：36,351 sequences/2,353 objects、source4 行/唯一路径对账和 `training_allowed=false` 均未变；无需重跑 full-file audit，reader/clock/provenance/geometry/QA/training gates 仍保持原状态。

- 05:02 CST direct-only source7/8 contract probe：source7（仅 dev）与 source8（仅 train）各取一个 canonical window，底层 `validate_sequence`/`pose_mesh_sample` 和现有 `egodex_data.forecasting_collate` 的字段/有限性检查通过，并在每源一个全有效窗口上复算 `point_valid`、`object_valid`、`goal_object_delta`；混合 pair 可 collate 但 local `sample_id` 碰撞，三项 provenance 丢失。脚本同时显式确认正常两边 split reader 对两源均 `REJECTED_AS_EXPECTED`，未改变 manifest、split policy 或 training admission；该结果仍是 bounded wiring smoke，未覆盖 invalid mask、完整 7/8 coverage、metadata/anchor 对账、clock/geometry。

- 05:10 CST live-index mtime 更新后再次做 5 秒 semantic drift probe；hash、source4 行/唯一路径、总行数/对象数与 `training_allowed=false` 仍稳定，未触发 full-file/geometry 重审计。

- 05:25 CST 监督器新快照后的 semantic drift probe 仍为稳定内容（36,351 sequences/2,353 objects、source4 对账 exact、global flag false），未触发 full-file/geometry 重审计。

- 05:40 CST 新监督快照后的 semantic drift probe 仍为 `STABLE_SEMANTIC_CONTENT`；未触发 full-file/geometry 重审计，所有 reader/clock/provenance/geometry/QA/training gates 维持原状态。

- 05:55 CST 监督器新快照后的 semantic drift probe 仍稳定（36,351 sequences/2,353 objects、source4 对账 exact、global flag false），未触发 full-file/geometry 重审计。

- 06:10 CST 监督快照仍为远端 stale 33,520/80,000、SSH 不可达；native index 36,351/2,353 且 global `training_allowed=false`，RL660 19,000 已完成，GPU 空闲、磁盘约 401.5 GB。watcher mtime 更新后的 5 秒 drift probe（`live-manifest-drift-0610.{json,md}`）仍为 `STABLE_SEMANTIC_CONTENT`，source4 33,056 行/唯一路径及总量对账 exact，未重跑全量文件/几何审计。

- 06:25 CST 新快照仍显示远端 33,520/80,000、SSH 不可达，native 36,351/2,353 且 global flag false；GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0625.{json,md}` 的 5 秒语义双读仍稳定，未触发 full-file/geometry 重审计。

- 06:40 CST 快照仍无状态转变：远端 33,520/80,000、native 36,351/2,353、global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0640.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，未重跑 full-file/geometry 审计。

- 06:55 CST 快照继续无转变：远端 33,520/80,000 且 SSH 不可达，native 36,351/2,353、global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0655.{json,md}` 语义双读稳定，未重跑 full-file/geometry 审计。

- 07:10 CST 快照继续无转变：远端 stale 33,520/80,000，native 36,351/2,353 且 global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0710.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，未重跑 full-file/geometry 审计。

- 07:25 CST 快照继续无转变：远端 stale 33,520/80,000，native 36,351/2,353 且 global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0725.{json,md}` 仍稳定，未重跑 full-file/geometry 审计。

- 07:40 CST 快照继续无转变：远端 stale 33,520/80,000，native 36,351/2,353 且 global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0740.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，未重跑 full-file/geometry 审计。

- 07:55 CST 快照继续无转变：远端 stale 33,520/80,000，native 36,351/2,353 且 global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0755.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，未重跑 full-file/geometry 审计。

- 08:10 CST 快照继续无转变：远端 stale 33,520/80,000，native 36,351/2,353 且 global flag false，GPU 空闲、磁盘约 401.5 GB。`live-manifest-drift-0810.{json,md}` 仍为 `STABLE_SEMANTIC_CONTENT`，未重跑 full-file/geometry 审计。

- 约 08:15 CST 生成只读 `cleaning-ledger.{json,md,png}`：source4–9 的 source-id 覆盖、行数对账、全局/逐源 training flag 关闭、source4 文件/语义复放及 source9 orphan 列表交叉核对均通过；所有 gate 状态与 failed/unverified 列表一致。该 ledger 仅提出并列 workstreams，仍明确 `ledger_is_proposal_only=true`、六源 `admissible_now=false`、`training_started=false`；不改 manifest/production loader，不执行 geometry/quarantine/clock/provenance 修复，也不启动混合训练。

- 约 06:00 CST 完成 `tmp/goal-supervision-20261015/mixed-collate-pair-matrix.{json,md,png}`：从 source4–9 各取一个窗口并覆盖全部 15 个不同源无序 pair，现有 `ptv3_shared_wam.egodex_data.forecasting_collate` 对每对都能正确 padding，保留 source 顺序，且 sampled `point_valid`/`object_valid`/`goal_object_delta` 与输入重算一致、浮点 batch 有限。所有首窗均为 local `sample_id=[0,3]`，三项 provenance 均被丢弃，说明 batch 身份必须用 `(source_id, sample_id)`，dedicated adapter 仍是 blocker。source4 只抽取 standalone 320-sequence index，source7/8 走 direct path；没有测试反向顺序、同源 pair、batch>2、scene-cap 截断或 invalid mask。接口与可视化子代理复核 JSON/MD/PNG 一致；这仍是 bounded wiring smoke，不改变任何 admission flag 或训练状态。

- 约 06:05 CST 完成 `mixed-collate-stress.{json,md}`：六源正序/反序 batch=6、HOT3D 同源重复 batch=3 均通过 source 顺序、派生字段和有限性；真实 HOT3D `[0,99]` 部分无效窗口与 source4/source7 混合时，fixed/padded masks 和 masked targets 全零保持。对同一 6-object/3072-point 窗口将 `scene_cap` 降至 512 后仅保留 object 0 的点而仍声明 object 0–5 valid，暴露单样本的 order-biased truncation 风险；默认 4096 cap 对该样本不截断。该结果仅收紧未来 adapter 合同，未改 manifest、准入或训练；接口与可视化子代理复核通过。

## 2026-10-15 native→v2 接口审查与准入 Decision Note

- 决策：继续保持所有 provisional native manifest 为 `training_allowed=false`，不启动多源混合训练，也不在未获架构确认前改写 v2 production loader。当前证据足以推进只读合同审查，但不足以形成训练准入或科学结论。
- 证据：`tmp/goal-supervision-20261015/native-contract-audit.json` 已按 manifest-listed 路径独立审计 source5/6/7/8/9，并检查 live source4 legacy geometry；五源 direct NPZ 的 shape/finite/30Hz/mask dtype 均通过。H2O manifest 524 个序列/67,358 帧，目录另有220个 orphan NPZ，已单列为 blocker，不再混入统计。
- 质量阻断：source5/6/8/9 与 source4 存在非单位或零/极小法线；DexYCB source8 法线整体约 `2π` 尺度，H2O source9 混有约 `2π` 文件；HOT3D source6 有55,361/999,865帧 `native_qa_pass=false`、19个非单调边、4,717帧残差超过2ms。多源行级 provenance、时钟确认、窗口 QA 过滤和对象 split policy 仍未完成。
- 法线修复仅做了只读 probe（`tmp/goal-supervision-20261015/normal-repair-probe.json`）：DexYCB 原始 OBJ 法线方向与独立重算的中位 cosine 为 0.9985–0.999997，支持“非零法线版本化归一化、零/极小项隔离”的候选策略；未覆盖 source orientation/语义，未改写任何 geometry。
- 接口证据：`tmp/goal-supervision-20261015/native-v2-contract-probe.json` 用真实 PoseMeshWindows+native collate 对 source5/6/9 各抽一个 TRAIN window；required v2 keys、T24/2×11/O×32×3 shape、v2 objective 及 tiny stand-in geometry 的梯度链路均 finite。旧 `shared_collate` 三源均明确以 `KeyError('points')` 拒绝，证明不能直接复用旧入口。
- 另做了 bounded semantic replay（`tmp/goal-supervision-20261015/native-window-semantic-replay.json`）：source5/6/9 的 TRAIN/DEV 各均匀抽 8 窗，anchor inverse、目标点 endpoint round-trip、invalid mask 零值、finite 字段及两样本 native collate 均为零误差；这是窗口合同复核，不是跨源语义或模型收益证明。
- 证据边界：该 probe 是单窗口 CPU plumbing smoke，使用 fallback identity stats 与 stand-in geometry；不证明生产 sparse geometry 的 voxel/feature 语义、混合 adapter、target semantics 或训练有效性。质量图为 `tmp/goal-supervision-20261015/native-contract-quality.png`，metadata/split 阻断图为 `tmp/goal-supervision-20261015/native-contract-admission.png`，均仅作审查辅助。
- 追加的 read-only admission dry-run（`tmp/goal-supervision-20261015/admission-dry-run.{json,md}`、`admission-gates.png`）把 provenance、metadata boundary、clock、geometry、file、QA、split、reader、training flag 九类门禁逐源列出；source4–9 当前均 `admissible_now=false`。reader gate 已要求明确成功 smoke（含 windows 与 `goal_roundtrip_max_abs≤1e-6`），未知、`SKIPPED`、`ERROR` 或 `REJECTED` 不准入；metadata boundary 对 source5/6/9 明确 FAIL、其余未复现源保守为 UNVERIFIED，收紧后的 JSON/Markdown/热图已由子代理复核一致。
- 接口边界的独立复现（`tmp/goal-supervision-20261015/native-v2-metadata-boundary.{json,md,png}`，详述于 `native-v2-interface-audit.md`）显示 source5/6/9 的旧 `forecasting_collate` 会丢弃 `source_profile`、`anchor_world_to_sample`、`training_allowed`；删除或伪造这些字段（含 `source_id=999`、`training_allowed=true`）仍被接受。子代理复核通过；这不是模型结论，而是 dedicated adapter 必须在 collate 前 fail-closed 并携带不可伪造 provenance 的 blocker。
- live index watcher 仍每分钟重写 manifest（当前约36,351 rows、source IDs 4–9）；row/path 对账可复算，但 source-level limitations/clock metadata 未逐源复制。旧 `shared-interface-audit-r1.json` 仍是 2026-10-14 19:24 的旧快照，不能代表当前 live index；冻结前不得引用其 `ENGINEERING_PASS` 作为准入证据。
- 独立 provenance/freshness 审计（`tmp/goal-supervision-20261015/live-provenance-audit.{json,md}`、`coverage-freshness.png`）对当前快照逐路径复核：36,351 sequences/2,353 objects 全部 source↔live exact，缺失/多余/非 path 改写均为 0；但 source-level `source_clock/git_commit/source_manifest` 等元数据未进入 live，且旧 audit 明确 STALE（train 差 +21,861、dev 差 −258，未覆盖 source5/9）。图和脚本由子代理复核，watcher 未停止。
- 监督更新（01:55）：远端状态仍为 stale step 33,520/80,000 且 SSH 失败；本地 RL660 geometry-lift 为 step 16,400，GPU5/7 约100%利用、显存约15.4/11.5GiB；磁盘余量约403.5GB。watcher/supervisor/live index 均存活，未停止或重启既有进程。
- 监督更新（02:10）：本地 geometry-lift 到 step 17,000/19,000；远端仍 stale 33,520/80,000 且 SSH 不可达；live index 仍 36,351/2,353、`training_allowed=false`。磁盘余量约401.0GB，仍高于20GiB保护线。
- 监督更新（03:02）：RL660 geometry-lift run 已自然完成 19,000 updates、exit code 0，GPU5/7 已释放；该 Probe 仍只记为工程 `UNCLEAR`，不改变 native 多源准入或启动混合训练。远端 stale/live watcher/supervisor 继续运行。
- 监督更新（03:10）：自动 supervisor 仍存活至 10:00；远端状态仍 stale step 33,520/80,000，live index 36,351/2,353 且 `training_allowed=false`，磁盘约402.1GB。
- 监督更新（03:16）：按 watcher 最新写入重新执行 provenance、native geometry 和九门 admission 审计；live manifest 读前后稳定，36,351 sequences/2,353 objects 对账仍 exact，source 读取无错误，旧 audit 仍 stale。native audit 与其后重跑的 admission dry-run 仍为 `BLOCKED`/全部 `admissible_now=false`（source9 orphan 220、HOT3D QA false 55,361/999,865、非单调边 19、残差超2ms 4,717）；子代理复核通过，未改写 manifest 或启动训练。
- 监督更新（03:25）：supervisor 新快照仍显示远端旧状态 33,520/80,000、SSH 不可达；live index 36,351/2,353、`training_allowed=false`，磁盘约402.1GB，GPU5/7 空闲。随后按最新稳定 manifest（03:27）依次重跑 provenance、native geometry、admission、split 和 clock probes；对账仍 exact，九门 admission 仍全部不准入，未修改 manifest。
- 只读 split/clock probes（同一 stable manifest snapshot）补充了清洗决策证据：[`object-split-policy-probe.md`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/goal-supervision-20261015/object-split-policy-probe.md) 显示当前 group split 在所有 source 都无交叉，但 source5/6/9 的 object↔group 图各自跨 train/dev 形成单一连通组件；强制 object-disjoint 会把三源 DEV 全部压为 0，隔离 crossing component 则整源无行，故没有自动重切。 [`clock-provenance-probe.md`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/goal-supervision-20261015/clock-provenance-probe.md) 显示只有 HOT3D source6 的 NPZ 带 native timestamp；source5/7/8/9 只有 frame-id 加转换器生成的 `time=index/30`，source4 仍需 legacy raw reader，不能把统一 `time` 当原生 30Hz 证据。
- 只读 target/window 语义审计（manifest stable snapshot 03:33–03:35）：[`target-semantics-probe.md`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/goal-supervision-20261015/target-semantics-probe.md) 显示所有 canonical rows 的 `target_slot` 均在范围内且与 `object_pose_valid` 维度相符，但 HOT3D target validity 仅 810,374/999,865 帧（每列 availability mask 完全相同），其中 QA-false 55,361 帧只是 target-invalid 子集，另有 134,130 个 QA-true 但 target-invalid 帧；source7 没有 selection metadata，source8 的 slot 分布需保留 native 语义。[`window-quality-probe.md`](/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/tmp/goal-supervision-20261015/window-quality-probe.md) 重放当前 stride=8 index：HOT3D TRAIN/DEV 只有 98.46%/98.13% 的已索引窗口 target horizon 全 valid，仍有 27/4 个含 QA-false 窗口；HO-Cap/DexYCB 只有约 0.5 的中位 future hand validity。该证据要求 adapter 同时实施 target-valid、QA 和 hand-valid 门禁，不能只依赖现有 reader 的 masked future。
- 只读 fail-closed adapter 原型（manifest stable snapshot `1792008512.9776218`）扩展到 sources 4–9：source4 使用 standalone 320-sequence UniDex compact index 的正常 windowed reader，source5/6/9 使用各自 windowed reader，source7/8 因单边 split 由同一 `validate_sequence` + `pose_mesh_sample` 做 direct canonical replay。六源首个 sample 均为 `VALIDATED_PROVISIONAL`（不是准入），缺字段、伪造 source/profile/training flag/sample_id、非有限或有限但错误的 anchor 均在 native collate 前 `REJECTED`。sidecar 仅由原型在外部暂存，现有 collate 会丢弃 provenance；独立扫描 audited manifests 的全部 sequence rows 均没有显式 `training_allowed` 字段，global flag 仍为 false。source4 replay 不覆盖 live 的 33,056 rows，也不验证 legacy clock/provenance；source7/8 direct replay 不豁免正常 reader 的 train/dev 门禁。
- 03:44–03:47 在相邻 stable live snapshots 上重跑 target/window、split、clock、provenance、native contract 与九门 admission：36,351 sequences/2,353 objects 对账仍 exact，source4–9 全部 `admissible_now=false`；HOT3D target horizon 全有效比例仍为 TRAIN/DEV 98.46%/98.13%，source6 QA/time blocker 与 H2O 220 orphan files 未变。新增完整 native audit 仍为 `BLOCKED`，未修改 manifest 或启动训练。
- 03:55 自动监督仍报告远端 SSH 不可达的旧 33,520/80,000 快照；live index 36,351/2,353、`training_allowed=false`，本地 RL660 已完成 19,000，GPU5/7 空闲，磁盘约 402.1 GB。
- 随后更新的 admission dry-run 优先读取五源 fail-closed probe：source5–9 的 metadata gate 由 `UNVERIFIED` 收紧为 `FAIL`（现有 collate 丢 provenance、global admission 仍关闭），source4 保持 `UNVERIFIED`；全源仍 `admissible_now=false`。
- fail-closed probe 扩展到 source4 后，admission dry-run 的 metadata gate 对 source4–9 均为 `FAIL`；source4 reader 仍 `UNVERIFIED`，因为 standalone replay 只覆盖 320 条而非 live 全量，整体仍 `admissible_now=false`。
- 下一步：等待用户对已提出的 source-profiled canonical record → admission manifest → dedicated native→v2 adapter 设计明确确认；在确认前仅继续审计、可视化和进程监督，不写 production adapter 或启动混合训练。预计资源成本低；若准入 blocker 继续存在，沿 quarantine/修复路径推进，不将工程 smoke 升级为结论。

## 2026-10-15 04:23 source4 全量 live 文件一致性 Decision Note

- 当前问题：source4 的 `file_consistency` 是真实数据缺陷，还是仅因 legacy reader 证据缺失而未决？
- 关键证据：只读流式审计 `tmp/goal-supervision-20261015/audit_live_source4_files.py` 对 live manifest 的 33,056 条 source4 行逐条打开 NPZ；33,056/33,056 文件、3,305,600 帧、33,056 唯一路径均通过，standalone/expansion 分别为 320/32,736，TRAIN/DEV 为 26,448/6,608，compact 数组/身份/reset 合同错误为 0。去除 watcher 易变 `checked_at` 后 manifest 指纹前后一致；16 个跨批次语义样本复放 16/16 通过。
- 行动与边界：把 admission dry-run 中 source4 `file_consistency` 从 `UNVERIFIED` 更新为 `PASS`，但仅表示 manifest-scoped compact 文件/身份合同通过；`reader` 仍 `UNVERIFIED`，clock/provenance/geometry/QA/training flag 仍独立阻断。抽样 geometry 和合成 validity/segment 的语义复放不升级为全量 reader 证明；不改 manifest、不修 geometry、不翻转 `training_allowed`，不启动混合训练。
- 成本/停止条件：CPU 文件流式约 3 分钟、无 GPU 和外部写入；若 live 内容指纹变化、任一身份/数组错误或语义样本失败，报告即失败并保留原准入状态。两名子代理分别复核脚本边界及 JSON/MD/PNG 一致性，未发现误升级。
- 随后针对唯一 source4 几何 blocker 做了 raw OBJ provenance probe：`tmp/goal-supervision-20261015/source4-piano-normal-probe.{json,md,png}`。Piano 对象 511 个采样点中 2 个零法线（均不在 32 个 query 点），对应 raw `decomposed_012.obj` 的同一局部面片；incident face normals 成相反方向并抵消，受影响为 TRAIN 的 16 条 live rows。最近邻法线的符号并不唯一，因此自动归一化/随意补法线不安全；probe 建议隔离该对象，直到源 mesh 重生成或完成方向审查。该结果只保留 geometry gate=FAIL，不改写 geometry/manifest/training flag。
- 04:42 的 live-manifest drift probe（5 秒双读）显示去除 watcher 的 `checked_at` 后语义 hash 未变化；总量仍 36,351 sequences/2,353 objects，source4 仍 33,056 行且 33,056 唯一路径，与 full file audit/native audit 对账一致。该 probe 只决定是否需要重跑全量审计，不提升任何 reader/clock/provenance/geometry/training gate。

- 监督（约 01:38 快照）：远端 WAM-v2 status 文件仍停在 step 33,520（SSH tunnel 当前不可达），本地 RL660 geometry-lift 已到 step 15,720；native live index 36,351 sequences/2,353 geometries，仍为 engineering-only。`tmp/goal-supervision-20261015/watch_until_1000.py` 每15分钟记录一次，未停止或重启任何既有进程。

- TACO 全量工程链路已完成：`Hand_Poses.zip` 15,231,790,511B、223/223 个 Range，连同物体位姿/模型包完整发布；2,317 个序列、363,198 帧、162 个几何对象转换完成，0 个转换失败。
- `taco-interface-snapshot-20261014-r1/provider-audit.json` 为 `ENGINEERING_PASS`：TRAIN 30,642 窗、DEV 7,946 窗；静态审查图为 `geometry-review-source5.png`。转换与审计产物均保持 `training_allowed=false`。
- `native3d-live-corpus-20261014-r1` 已追加 TACO source_id=5（TRAIN 1,826、DEV 491），当前总索引为 36,351 个序列、2,353 个共享几何；统一 mixed audit 的结果仍只作为工程合同检查，不是训练效果证据。
- TACO 原生 30Hz 时钟、目标选择和跨源语义仍未准入；正式训练前需冻结索引并完成 source admission。Task 合同测试当前 17 项通过。

下方 2026-10-14 条目保留为过程与决策记录；若与实时产物冲突，以本节和 `status_geometry.py` 输出为准。

## 历史快照（2026-10-14 22:55）

- HOT3D、H2O、UniDex扩产、HO-Cap复用、DexYCB既有subject10均已完成工程转换；DexYCB不是全体subjects。UniDex共2066mesh/33056条执行，82/82批与逐批审计完成。
- HOI4D手cache1858段/458791帧完成；rigid手物共同坐标输出1326段/397800帧/286 CAD（TRAIN1062、DEV264），0实现失败。194段TrashCan多部件对象明确跳过，失败历史保留。完整native审计通过，camera*object复算最大差1.58e-7；359840帧手标签、397784帧物体标签有效，缺失帧保持mask。
- HOI4D仍不接入统一30Hz索引：1251段与官方名义15Hz相符，69段duration与decode冲突，6段无可判定duration。6段bbox/CAD比例明显偏离，单独列入native-bbox-review-cases.json，不自动拉伸mesh。静态图native-geometry-review.png可人工查看；全部仍training_allowed=false。
- TACO仅手部ZIP未到齐，22:53记录200/223个64MiB分片，最近平均约0.29MB/s；这是网络transport瓶颈，GPU6尚在等待。远端22023隧道不可用，远端下载未启动；已向用户说明恢复方法，本地4线程续传继续。

## TACO等待与恢复 Decision Note

- 决策：避免下载耗尽转换期限，仅替换已核验UID/cmdline且处于WAITING_NATIVE_ARCHIVES的本项目converter。输入等待最多24h，下载完成后另计2h GPU6转换；转换前重新检查GPU占用，保留六卡上限和20GiB空间余量。
- 本地supervisor最多4h，每30秒看本项目状态，仅在旧writer已退出且为PARTIAL/STOPPED时最多重启4次4线程、各1h下载；不因远端隧道缺失中断健康本地writer。supervisor不自动切远端；远端helper先SSH认证，成功后才请求本地停止，已通过断线不触碰本地状态的回归测试。
- 输入到齐后自动等待转换、冻结TACO manifest并执行全量窗口接口审计和静态图；转换未遍历全量时记录PARTIAL，失败序列保留ledger，不写成全部完成。下载与转换子进程各有独立期限，supervisor到期不杀仍在预算内的子任务；STOP只停止对应管理循环，下载STOP在source目录、转换STOP在converted目录。
- 成本：本次仅CPU监督与网络重试，不新增训练；真实221帧GPU reader smoke完成，3项断流/远端预检测试通过。运行receipt记录实际commit、PID和参数。

下文为此前阶段记录，若状态冲突，以本节和实时progress为准。

## HOI4D原生手物合并（2026-10-14）

- GPU手缓存已完成1858段/458791个native frame ID；修复一条poseCoeff为标量的原始记录为invalid mask，保留原帧和失败历史，不丢弃整段275帧。
- 3段真实原生pose smoke通过，raw LOG与物体RT共同坐标复算误差小于1e-6；14项合同测试通过。仅处理rigid CAD类别，保留canonical原点/米制；bbox dimensions作审计字段，不作为每帧scale。
- release.txt2971行、testset.txt933行实际有31个重叠ID；原生合并只使用release且不在testset的序列，优先防泄漏。按全局<C,N>实例分TRAIN/DEV，未准入物理30Hz窗口。
- 当前批处理：CPU流式读取已下载ZIP并合并已GPU生成手部cache，单worker4CPU线程、2h、20GiB余量；纯标签/矩阵/文件处理不额外占GPU。保留JSON内部frameId偏差、原生相机/物体矩阵、clock conflicts，输出单独native schema。失败留ledger，缺失mask不补假静止。

## TACO远端残余分片 Decision Note（2026-10-14）

- 证据：本地下载反复超时/断流，当前约0.25MB/s；远端直连8MiB实测3.915秒、HTTP206/精确Content-Range，约2.14MB/s。下载而非GPU计算决定剩余时间。
- 行动：先在STOP安全边界停止已核验归属的本地range writer，保留全部completed和partial；只将残余不超过4GiB分片在既有授权远端workspace补齐（4个CPU网络线程、单请求8MiB、1h），通过固定SSH host key拉回并逐段合并本地原archive，不重复全部15GB。
- 边界：URL仅经SSH stdin进入内存、不落日志/清单；不占远端GPU、不触动已有训练、不创建身份。双方20GiB物理余量，deadline或STOP停止，已完成数据保留。补齐后原GPU6 converter自动开始；残余transport失败保留partial并报告，不承诺短探针速度会始终维持。

## 原生HOI4D GPU处理 Decision Note（2026-10-14）

- 授权/资源：用户要求继续处理并允许使用空闲GPU。已核验本地4/6空闲、5/7和远端0/2为既有训练；HOI4D用GPU4、TACO保留GPU6，全部项目最多六卡。
- 决策/证据：先缓存全部1,858段手部原生MANO/FK到相机系21点，保存原生frame ID、参数、2D重投影和缺失mask；3段真实GPU smoke及原始投影检查通过。随后按真实文件名把rigid物体RT、米制CAD和raw LOG camera_to_world接入原生序列。
- 实质限制：官方decode固定15fps，但300帧对应action.duration10s/20s均实际存在；当前不假装全量30Hz、不静默插值，原生缓存与未准入pose数据先独立保存。内部frameId含关键帧编号，配对只使用文件名数值，原字段保留审计。CAD不减顶点均值、不盲乘bbox尺寸。
- 成本/停止条件：GPU4单worker、CPU4线程、2h期限、20GiB物理余量、STOP序列边界；不生成视频、不改已有训练。失败序列留ledger，不把cache误纳入统一30Hz训练集。TACO残余下载用已实测更快的远端HTTP线路补齐后再GPU转换。

# 原生3D数据接入状态

## HOI4D Windows 取数 Decision Note（2026-10-14）

- 授权/证据：用户明确授权按`docs/user/连接远程windows.md`从`D:\BaiduNetdiskDownload`取数，并已启动反向SSH隧道。已用现有服务器密钥连接oyx@127.0.0.1:2222，四个ZIP大小与官方清单完全一致，共24,189,914,732B。
- 行动：只读SFTP拉取camera/hand/CAD/annotations四包至`hoi4d-native-full-20261014-r1/archives`；Windows源文件保留。2并发、断点partial、严格固定本次SSH host key、完整长度和ZIP目录检查后原子发布；不传用户其他文件。
- 成本/停止条件：CPU/网络、最多24h、20GiB物理余量，STOP文件可停自己的SFTP子进程并保留断点；不会修改Windows系统设置或源数据。
- 初步实测：hand490,584个pickle/1,858个sequence ID，左31,845、右458,739；annotations远端只读目录891,900 objpose、2,973 camera trajectory/action。左右各一帧FK投影与原生kps2D平均误差约0.7px；已实际看到文件名00000与内部frameId1不一致，配对必须另核。
- 传输已完成：4/4包、24,189,914,732B、824.5秒；长度及ZIP目录检查通过，收据在该输出的download-summary.json。Windows源文件未改动，隧道不再需要为本次传输保持开启。
- 下一步：输入先检查hand/objpose/camera轨迹及rigid类别的共同坐标与原生时钟；annotations中的segmentation不自动当pose训练输入。

TACO网络恢复补记：原下载遇到transport断流，保留已完成分片。用真实HTTP故障服务复现了64MiB分片中途断流后重试从头丢失已接收前缀的问题；修复为每次请求不超过8MiB且同一分片重试保留已确认字节。该故障测试先失败后通过，Task共9项测试通过；STOP仅用于核验归属的旧writer安全切换，不丢弃完成分片。

## TACO GPU6 加速 Decision Note（2026-10-14）

- 决策：用户询问增加GPU并行后，启用空闲本地GPU6执行TACO MANO FK；保留GPU4上的UniDex队列，余下约3–4批不为临时并行重排任务。
- 证据：已核实本地5/7和远端0/2是既有训练，UniDex用本地4，共5卡，增加6不超过六卡上限。真实221帧标注及仅用于测速的8倍平铺输入，FK在GPU快3.19/3.55倍；全reader GPU smoke与CPU原生数组差≤7.45e-8m，object/time和mask不变。数据解码和写盘仍在CPU，不能按FK倍数宣称全流程提速。
- 行动与成本：先确认旧TACO converter仍处WAITING_NATIVE_ARCHIVES、UID/cmdline归属正确，再替换为GPU6版本；原生ZIP完成后自动开始，转换前再次检查该卡无其他占用。CPU4线程、单GPU、2h期限、20GiB物理余量，无视频，STOP可在序列边界停止。
- 时间边界：21:13盘点TACO下载约余20分钟；2317序列的全量转换吞吐尚未测，暂估15–40分钟，后续审计约5–15分钟。该预算是工程估算，开始转换后用实际完成速率更新。HOI4D等待用户四包，DexTrack缺官方可读输入，均不计入可承诺完工时间。

## H2O 接入 Decision Note（2026-10-14）

- 决策：用户授权凭证下载后，直接接入pose包已有的原生21点手、物体RT和逐帧cam_to_world；不新增视频估计或MANO拟合。
- 证据：实际subject1包同时含hand_pose（128数，左64/右64）、hand_pose_mano（124数）、obj_pose_rt和cam_pose；CAD米制范围约3–26cm。官方player以cam_to_world乘物体RT，官方pose split按session分开。
- 行动：统一右/左11点，保留原始21点、MANO、相机轨迹和action/verb；仅使用cam4，官方train/val进入工程索引，subject4 test写独立heldout-test-manifest，不进入自动混合索引。原始帧号断点拆段，短于28帧不造窗。
- 成本/停止条件：CPU流式读取每份tar，不完整解包近百万txt；2h期限、20GiB物理余量、STOP文件边界可停。CPU用于纯标签转换，无学习模型；不新占GPU。相机/物体非法旋转或缺失帧不补假标签。
- 限制：论文记载30fps采集，但当前原生包没有逐帧时间戳，clock仍待准入；原生joint profile与MANO重建profile分开。工程转换不改变training_allowed=false。

## 2026-10-14 全量扩产 Decision Note

- 用户明确取消产物磁盘空间上限并要求立即推进；根CAMPAIGN据此更新，六张项目GPU上限与外部只读/身份边界不变。
- 行动：HOT3D全部ground_truth/hand_data/assets约2.70GB；TACO三份几何ZIP约15.69GB，CPU下载与转换并行；复用官方资产启动UniDex新增mesh采集，随后HO-Cap轻量接入。
- 输入：用户三个授权清单只读；TACO/HO-Cap官方公开几何包；UniDex官方TRAIN中排除TEST mesh，保留train/dev按mesh分组。源数据不因未来结果筛选。
- 成本与停止条件：HOT3D下载4线程、TACO手部ZIP8线程独立Range、24小时可恢复期限，单次转换内存有界；UniDex实际每批25对象/200env/两轮（末批更少；原collector最多256env）、GPU4单worker、每批≤10分钟；扩产输出只受实际文件系统剩余≥20GiB约束，不生成视频。既有训练与未知进程不动。
- 结果仍属工程语料，不自动准入训练；缺权限或临时URL失效及时向用户说明。源clock/语义问题保留明确mask与provenance，不伪造标签。

## 扩产实际状态（初始样例之后）

- HOT3D849包已全部完成：2,701,783,954B压缩、9,069,098,557B展开；Aria136+Quest158=294个记录提供手物GT，130个不提供。290段转换为999,865个名义30Hz时间步，4段没有QA通过的原生时段跳过，0实现失败。
- 全序列版保留native/grid timestamp和最近邻残差，容忍≤16ms；重复选中的native帧标invalid。它不是插值后的额外真实观测，也不是正式时间精度准入。原512步pilot仍保留其≤2ms规则。
- UniDex排除已有20mesh和官方TEST后，2046个新mesh分82批，每mesh16条真实执行，正在GPU4持续采集与自动审计。原20保留，总目标2066mesh/33056条；实际完成数看progress，不写成已经全量完成。
- HO-Cap旧Task已有9人64段72944帧，经原作者MANO与标签检查；只读复用成64条DEV工程序列，继续保留外部测试用途。DexYCB已有subject10，100段7282帧直接读取一个固定相机的joint_3d/pose_y，不处理图像、不重复8个视角；遵循官方s1把subject10全部放TRAIN。两者native clock仍明确未核实。
- TACO物体位姿和模型包完成，15.23GB手包并行Range续传；完成后等待中的converter自动按序列读取ZIP，避免再全量解包26.99GB pickle。真实pilot重打包验证全量reader，全部原生数组与pilot一致（1e-7容差）。
- `native3d-live-corpus-20261014-r1`每分钟自动追加完成分片、共享CAD引用，不把已有pilot和full重复纳入。5项合同测试通过，混合来源shared-collate已通过工程检查；没有开始多源模型训练。
- 旧393/400GB全局watcher已核验归属后替换为纯容量/远端训练监控，避免扩数据误停已有训练；相关receipt在项目tmp。
- H2O用户提供的HTTP Basic凭证已实际验证：六包均HTTP200，合计326,238,749B；已启动4线程可恢复下载（1h期限、20GiB物理余量），凭证只存本机mode0600文件，不进入Git或日志。手/物体pose和CAD下载后再核验坐标、时钟、split；下载成功不等于训练准入。HOI4D仍需网盘下载访问，DexTrack官方受限且社区镜像有混合修改lineage，暂不把kinematic references记作执行数据。

H2O新增实测：138个TRAIN/DEV原始take已变为524个连续片段、67,358帧，8个CAD；22个TRAIN帧所在连续段短于28帧而跳过。全量6,871窗口接口审计通过（TRAIN5,691/dev1,180），真实原生21点与官方MANO约定spotcheck最大差8.14e-8m；8项合同测试通过。静态图与冻结审计位于`h2o-interface-snapshot-20261014-r1`，多源冻结接口检查也通过；H2O六包现已全部完成，含subject4 test的总转换为184个take/744个连续段/90,749帧，0失败；test220段/23,391帧独立保留，未进入多源索引或训练。

实时查看：`python3 src/task/3d-data-preparation/tools/audit/status_geometry.py`。输出与日志在`outputs/3d-data-preparation/`和`tmp/3d-data-bulk-20261014-r1/`。下文初始512MiB/400GB预算是历史快照，已被本节用户授权覆盖。

## 2026-10-14 路线切换 Decision Note

- 决策：按用户授权暂停视频重建，转向已有连续3D手、物体pose/mesh的数据；不改变根MISSION的目标或claim。
- 证据：EgoDex首版每尝试窗约2.63秒且目标语义仍有问题；已有UniDexGrasp++20对象/320条执行的3200窗合同审计通过。
- 行动：核验归属后，用既有SIGUSR1安全边界暂停远端EgoDex下载器及GPU1/3旧重建，保留原始输入和全部证据；不停止已有模型训练。
  在本Task实现统一PoseMeshWindows，先把现有UniDex轨迹以按需方式接入，再取得TACO/HOT3D最小授权几何样例。
- 成本：当前实现/文件统计使用CPU，无新增GPU任务；原生样例下载初始≤512MiB、处理输出≤1GiB、临时输入/缓存≤1GiB。
  全量解包/回放和训练另按容量核验安排，遵守全局400GB及六卡上限，窗口不重复展开存储。
- 停止条件：输入/单位/时间/手与物体坐标不明确先停该reader；没有授权下载清单不代用户申请身份或接受条款；不可读数据/目录/资源边界及时询问。
- 下一步：source/profile明确、多对象当前anchor与缺失标签合同通过后，做真实模型工程接线；正式微调须另核验统计和来源采样。

外部访问：用户已提供HOT3D Aria/Quest/Assets授权清单，首批原生样例下载完成；三个清单保持本机且被gitignore排除。HOI4D分享读取失败、H2O需身份申请，尚不启动。

## 已完成的原生3D工程链路

- 视频下载和两条旧重建已正常STOPPED；保留全部数据，既有模型训练继续。证据：`outputs/3d-data-preparation/route-switch-20261014-r1/remote-inventory-r1.json`。
- `PoseMeshWindows`支持共享CAD、多对象、当前目标物体固定anchor、原生缺失mask、按需H4/K24；成功与失败执行均保留，控制量不混入人手几何标签。
- 现有UniDex20对象/320条执行接入新manifest，无复制原始数据，TRAIN2560/dev640。全部窗口与旧reader逐字段比对，最大差2.98e-8，并进入既有forecasting_collate。
- HOT3D两段授权真数据+完整Assets：下载177,089,578B，解压236,321,352B。各取512个时间步，TRAIN57/dev58窗、最多6对象、左右手；官方QA掩码实际应用。
- Aria选214-1时间流并最近邻对齐30Hz，偏差上限2ms，实测最大1.685ms；Quest最大约1ns。原始/grid时间戳及偏差保留，不插值pose、不把多相机时间戳当额外动作。
- 5项独立合同测试通过，覆盖固定anchor、法线/导数、多对象、未来干预、缺失标签、reset和共享collate。
- TACO一段下载完成：24.03MB成员payload、34.35MB原生输入，221帧双手/两物体生成25个TRAIN窗；新增compact序列和两几何仅99,947B。mixed-r2复用HOT3D后为TRAIN82/dev58，共140个新窗口。TACO30Hz仍是待证实的clock假设，不能正式训练准入。
- 数据继续标记`training_allowed=false`：目标操作语义、人手点profile、跨源normalization/sampler和模型工程接线仍未准入；两段HOT3D的参与者/设备划分仅工程使用。

空间与后续安排见`docs/user/架构.md`。初始下载/缓存边界不自动扩到全量；扩大采集前记录下一条Decision Note并复核全局400GB。
