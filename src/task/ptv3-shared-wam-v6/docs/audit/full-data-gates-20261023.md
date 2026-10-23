| 观测 | 含义 | 决定 |
|---|---|---|
| 八项全部通过，且生产 recipe、目标机器与输出路径绑定同一数据身份 | 数据面就绪 | 才排远端4卡九源大训练，另冻结配方与效果判据 |
| 任一项未过 | 必须指出具体缺口与修复估计 | 大训练继续 HOLD；只处理下表阻塞 |
| 抽样发现实际监督内容问题 | 对应数据不可信 | 回到转换/准入，禁止开训 |

# 2026-10-23 全量数据八门独立复核

**结论：还不能开大训练，HOLD。** 本地全量可读索引和内容抽样已过；**生产全量 recipe 未接入、远端冻结数据未落地、source8 DEV 缺失**没有关闭。goal证书计数已清楚，但合法goal对照仍没有冻结的配对划分。当前材料不构成训练排期。

这里“独立”指脱离 producer 自报计数，重读实际数组，并用 SciPy SE3/暴力距离复算内容；不是另一位人工审阅者背书。**实测**为全量索引/文件统计/实际前向；**抽样**为288窗口和1.5124%新固定文件；**既有记录**为此前去重进度与另一会话的远端32条转换探针；**估计/方案**均另标。

| 门 | 本轮证据与裁决 |
|---|---|
| 1 reader | 全量索引/shard逐行 PASS；生产 `V6Windows` 接入仍 HOLD |
| 2 内容/split | 母序列隔离、288窗及独立contact复算 PASS；九源 DEV 覆盖未过 |
| 3 身份 | 数据/ordered tuple/审计seed/smoke config固定；生产训练recipe/config尚不存在；执行源码记录限制见下文 |
| 4 磁盘 | 本地预算不通过；NAS容量/写权限通过，完整数据与配方落地未完成 |
| 5 覆盖 | 8个零格已逐格定位；仅可接受有边界的稀疏覆盖，不宣称108格齐全 |
| 6 goal/事件 | 旧6k=0，全量legacy=0，新采26,304个goal合法；意图事件证书仍0；合法goal对照配方HOLD |
| 7 一致性 | 本地与brief47同一个manifest，增删均0；远端原始pack不是逐数组同一身份 |
| 8 E2E | 实际全量适配reader→真实PTv3→监督loss PASS；不等同生产launcher/DDP已验收 |

## 1. 全量reader逐项对账（实测）

新增 `ptv3-shared-wam-v6/src/ptv3_shared_wam_v6/prepared_reader.py` 的 `PreparedWindows` 是**数据审计适配层**，沿用真实 `sample_window`/collate、joint21/H4/K24/2048几何/监督代码，强制unknown goal/no-event，按计划sequence_id寻址。它没有优化器采样器，也没有授予training_allowed。

重读**39,527/39,527个窗口shard**、全部11,197,332个ordered tuple，并逐行对比reader mmap；逐母序列检查offset连续、计数、tuple唯一、slot/tick范围和split。**没有因clock mask过滤、截断、抽样索引或用旧1,177序列替代全量。**

| source | TRAIN 读到 / manifest | DEV 读到 / manifest | 差额 |
|---|---:|---:|---:|
| 0 OakInk2 | 5,412,663 / 5,412,663 | 971,478 / 971,478 | 0 |
| 1 GRAB | 287,955 / 287,955 | 25,479 / 25,479 | 0 |
| 2 ARCTIC | 370,362 / 370,362 | 49,930 / 49,930 | 0 |
| 4 Shadow replay | 1,930,631 / 1,930,631 | 482,384 / 482,384 | 0 |
| 5 TACO | 238,297 / 238,297 | 61,408 / 61,408 | 0 |
| 6 HOT3D | 578,181 / 578,181 | 137,899 / 137,899 | 0 |
| 8 DexYCB | 2,953 / 2,953 | 0 / 0 | 0 |
| 9 H2O | 44,110 / 44,110 | 9,100 / 9,100 | 0 |
| 10 HOI4D | 475,898 / 475,898 | 118,604 / 118,604 | 0 |
| 合计 | 9,341,050 / 9,341,050 | 1,856,282 / 1,856,282 | 0 |

TRAIN31,833母序列，DEV7,694。[逐母序列对账](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/all-sequence-reconciliation.jsonl)记录每项offset、窗口数和字节；[完整reader收据](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/full.json)耗时255.83s。

**边界：**这是全量索引/shard读取，并非把1,119万份高度重叠点云全部物化。实际点云/监督消费为288分层窗和九源模型smoke。原生产 `V6Windows` 对这份审计manifest实测拒绝：`explicit admitted v6 event recipe required`；没有把拒绝改成绕过准入，也未修改生产训练入口。

## 2. 内容与split（全量隔离实测 + 288窗抽样）

全量检查 `(source_id, source_sequence)`、解析后的手部路径、存在的原始URI和完整手部payload identity：跨TRAIN/DEV交集**0**。同母序列不跨split。此证据不等于已经完成跨母序列近重复内容去重；旧summary仍被STOP中断，最终去重索引/catalog未完成。

抽样seed20261023，九源各32窗：每源TRAIN/DEV各16，source8只有TRAIN32；共**288个不同母序列、288窗**（TRAIN160/DEV128）。先均匀母序列，再均匀准入窗口。[逐窗dtype/shape/非零计数和tuple](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/sample-windows.json)可复查全部样本。

| 字段/检查 | 抽样结果 |
|---|---|
| pose | H28/目标slot的4×4 float32姿态通过有限值、SE3和有效性门；SciPy独立变换复算hand/scene监督，容差2e-5m |
| hand监督 | `(24,2,21,3)` float32；非零元素83.3549%，valid83.3550% |
| contact | `(1,24,2048)` float32；非零34.7149%，有效mask94.1406%；232/288窗口存在非零contact |
| part | 同shape int64，非零53.5410%；range/validity通过 |
| raw event | NONE-valid103、CONTACT-valid64、RELEASE-valid112、UNKNOWN9；integer type/delta、boolean valid，时间非负 |
| raw未来goal标签 | 288窗均非零，query delta元素非零率均100%；只作为标签事实 |
| 模型goal/event输入 | delta/points/type/time均规范零，valid均false；不是数据丢失，不把未来endpoint当合法条件 |

更强的独立内容检查：不调用writer的`surface_contact/cKDTree`，对288窗的current/+12/+24帧用**暴力cdist**重算surface→object距离、confidence和part：**811有效帧完全一致、最大contact误差0；53无效帧mask/零标签正确；最近点tie例外0**。[几何复算收据](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/geometry.json)。

时钟：抽样中有packed native `time/segment`的source4/5/6/8/9/10均实读并检查严格递增、30Hz和不跨segment。source0额外通过SSH只读重读原始**568条frame_ids**，严格递增/stride4及既有坏边集合一致；source1/2继承native120Hz stride4/native30Hz帧索引合同，**没有伪造绝对时间戳**。[source0实际原始时钟复读](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/source0-native-clock-reread.json)；缺绝对reset前可用时间是goal认证缺口，另计。

**source8 DEV=0的判断：**可明确标为“九源TRAIN、八源DEV”的受限拟合数据；**不能接受为九源泛化/九源DEV模型选择证据，也不能默认为本次完整放行已过**。应恢复官方DEV母序列并重做新manifest，不从TRAIN按物体身份放宽准入或挪成DEV。当前未改变旧split。

## 3. 身份与字节（实测 + 1.5124%抽样）

全量manifest SHA256：`4e40b4ec0bd58d390f8d5cee6047530d1832c3d5acb6080232251236d934c4c3`。本轮没有重写manifest，所有六个绑定文件重新SHA复核一致：

| 文件 | SHA256 |
|---|---|
| `dev-all.npy` | `3e24cbabd8e26d71addf33d3e874af0b85fea09ab4f4f18f2d8f9488a7b9afed` |
| `dev-clock-valid.npy` | `2c816f6b30b9afc8e652ff6170ed314ce07ea241a8de77e0dad20ab5008517ab` |
| `phase-geometry-source.csv` | `17e8735545012c17f63fe953de0843dc60ca880df32f7e292dc412064152d6d2` |
| `sequences.jsonl` | `98f800bbcec5cf57f99903cb4597c0c24f207e018f2c26029997ba79201b3f00` |
| `train-all.npy` | `b8d0314aeada83ad6878d5781d677b28e1a76e279cf4bb6107a7ac2744e1ea99` |
| `train-clock-valid.npy` | `b89d55692f07b1160e4cd6b099f1b9dfb631cf2f8d58905d6de9bcef8b97b5ee` |

旧6k manifest仍`5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`；旧ordered tuple仍`d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7`（原launcher的紧凑排序JSON，无末尾换行）；512锚点文件仍`440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706`。

smoke config SHA：`127cefbaf49370e18f54524202ef5224d77d2d3866c50778ecfeb0f8a8f45484`；原TRAIN-only stats SHA：`2bd4f7e553fc2e3ddffc0856b078790e4177aaae362d7ec6626d6a9ef3485d61`。seed集合为audit20261023/model170/DEV171/flow172/继承surface20261019，显式序列化SHA：`a63744dbfc8505076b0baf013c68e0b5e569f428d4c25635905314a29edb80a0`。[身份锁](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/identity-lock.json)同时固定seed字段、格式、global seed ledger和新采candidate manifest。没有把这些审计身份充作尚不存在的生产全量recipe/config。

**182,331,193,978 B三方精确一致**：38,350新转换×22个固定文件，共843,700文件，实际stat求和=逐receipt actual=expected。另1,177序列只读复用，未算进“新固定转换”字节。118,581个hand/contact/event metadata canonical identity与receipt一致；39,527个window shard SHA均一致。[完整磁盘逐文件大小](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/all-fixed-file-stats.jsonl)。

按source分层抽580个新转换母序列的全部固定文件，**12,760/843,700=1.512386%**，实际hash2,808,837,740 B。hand/contact的数组payload SHA与发布时metadata摘要对比，事件从contact重建且NPY字节SHA一致，metadata canonical SHA与原receipt一致。每个原始文件字节SHA也登记并即时复算。**没有历史JSON字节SHA的文件只声称canonical内容身份吻合，不伪造旧字节级基线。** [逐文件SHA](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/sample-file-sha256.jsonl)、[抽样汇总](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/hashes.json)。

执行源码的限制也保留：r2工具在**收据写出时**记录源码SHA，full运行期间新增了独立geometry stage，所以其终态SHA包含未参与full计算的新函数。已保存真正加载的base/geometry源码快照及版本解释，未覆盖原收据：[执行源码更正说明](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/execution-source-resolution.json)。所有数据文件身份重核一致；goal-r3代码在`188bebd2`运行。后续生产run必须在启动时固定代码与配方；工具已修为启动时捕获代码SHA/commit，不沿用这项记录缺陷。

## 4. 磁盘和输出方案（空间/小探针实测，训练预算是估计）

本次快照`/home2`普通用户可用**72,077,053,952 B / 67.127GiB**；保留30GiB，且没有为其他run未来写入重复领取空间。本轮两个审计attempt约188MiB，未删历史数据、checkpoint或失败产物。当前不排本地大训练。

沿用模型结构，现有真实checkpoint大小**1,217,116,001 B**。40k更新、每500步保存：80周期+step1=81，另给step20/128最佳选择2次和异常停止1次，**最多84次**。本次方案保留全部新checkpoint，避免依赖删除历史产物节省空间。

| 未来写盘项目 | 预算，未执行 |
|---|---:|
| checkpoint：84×1.5GiB上界 | 126GiB |
| stdout/train/dev日志与小评估JSON | 2GiB |
| 原子保存临时文件 | 1.5GiB |
| manifest/metadata额外余量 | 2GiB |
| 峰值合计 / 配置硬帽 | 131.5GiB / 136GiB |

输出方案：**远端NAS** `/mnt/ugreen_nas/storage/oyx_storage/wam-nine-source-full-20261023-pending/training`（提议路径，未创建训练run）。NAS实测可用**51,657,744,318,464 B**，约46.98TiB；本轮专用audit目录4MiB文件创建、写入、fsync成功。训练artifact硬帽136GiB、NAS低于30GiB停止新增写入；模型、缓存、TMPDIR均应在NAS，不在远端系统盘或本地`/home2`产生大件。

SSH关闭压缩的4MiB上传+fsync实测9.591s，**0.4373 MB/s**。按这个**小探针外推**，只传182.331GB约**4.83天**；按此前0.3–0.45MB/s区间约4.69–7.03天。还需复用依赖约6.008GB、registry约96.13MB、索引和协议开销，不能把4.83天当交付承诺。复制不会释放本地空间。未启动数据迁移。

另一会话的远端32母序列probe记录：32/32准入、24.56s、NFS写约12.20MB/s；这是**既有探针记录**，不是本轮全量实测，也不代表全量可在1h重建。远端输入文件齐全但存在下节版本差异；若在NAS原位重建以避开WAN搬运，必须先修正输入身份/边界绑定，再核对完整tuple和内容，不能直接将速度probe升级为训练准入。[完整空间/输出/带宽方案](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/storage-plan.json)。

## 5. TRAIN剩余8个零格（全量实测）

| # | phase | 几何 | source | TRAIN窗 |
|---|---|---|---:|---:|
| 1 | approach | elongated | 8 | 0 |
| 2 | approach | compact | 8 | 0 |
| 3 | contact | flat | 8 | 0 |
| 4 | contact | elongated | 8 | 0 |
| 5 | approach | flat | 10 | 0 |
| 6 | close | flat | 10 | 0 |
| 7 | contact | flat | 10 | 0 |
| 8 | hold | flat | 10 | 0 |

source8前4格是当前DexYCB准入样本/phase稀疏；source10后4格是本次TRAIN的flat几何类缺失。**受限于当前支持域的训练可保留这些0**，不插值、不从unknown搬数、不为物体放宽准入；不能声称108格完整覆盖。当前无需为了unknown-goal基础训练盲目新采。若研究claim要求这些格，则先查同source既有合法原始数据/几何，再决定有针对性的恢复或采集；v49 source4只能补source4，**不能填source8/10的格**。

## 6. goal与事件认证（计数和证据边界）

| 数据分区 | 母序列/rollout | 窗口 | 可认证goal | 可认证独立意图事件 |
|---|---:|---:|---:|---:|
| 旧6k TRAIN | 1,064 | 6,000 | 0 | 0 |
| 本次全量legacy TRAIN | 31,833 | 9,341,050 | 0 | 0 |
| 本次全量legacy DEV | 7,694 | 1,856,282 | 0 | 0 |
| v49 source4新采，未并入legacy | 96 | 26,304 | 26,304 | 0 |

全量legacy九源逐类均0：s0=0/6,384,141，s1=0/313,434，s2=0/420,292，s4=0/2,413,015，s5=0/299,705，s6=0/716,080，s8=0/2,953，s9=0/53,210，s10=0/594,502。不是所有字段为0，而是没有可绑定的完整事前intent/reset/availability证据。旧原始字段缺证依据本轮重新固定SHA的全量native census；未声称再次重建全部旧原始生产链。

新采96条逐条重新验证instruction签名、原始intent文件SHA、跨execution独立来源、available_at < reset、reset先于首观测、geometry/query绑定；**所有26,304个当前pose→合法task-goal编码都检查有限，192个首尾窗口经过实际 `reset_bound_sample` 消费**。合法角色计数为reset task instruction 26,304；event仍UNKNOWN_NO_INDEPENDENT_EVENT_INTENT 26,304。接触派生CONTACT/RELEASE标签不自动成为独立事前事件计划。[新采goal复核收据](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/../data-gates-20261023-r3/goal.json)。

**合法goal对照不能按现有生产配方马上开做。** 新96条只有official TRAIN assets、split仍unassigned candidate，尚无绑定同tuple/同初始化/同监督的unknown-vs-legal配方与独立DEV。可以据此准备一个**goal-only**有边界的对照设计；不能称已有九源合法goal或合法event全量对照。无goal/no-event的A训练不要求把旧数据补认证，但仍受reader/recipe/存储/DEV门约束。

## 7. 与brief47全量的一致性；远端差异的影响

本轮消费的就是 `full-rigid-20261023-brief47-r1/readiness-manifest-r1/manifest.json`，SHA未变。逐项差异：**新增母序列0、删除0、新增窗口0、删除0、TRAIN/DEV重新分配0**；全部ordered shard对账一致。96条新采未并入，没有把39,527写成39,623。未发布另一份“最终去重manifest”。

独立补查远端原始输入时，既有10个HOT3D样本的`segment`数组不同，不能宣称NPZ数组身份一致。为判断它是否污染准入，进一步读取**全部287条**远端实际segment/time，检查本地**716,080窗**：时间坏边集合差异0、segment边界不同10条、**跨远端边界的准入窗0**。十条具体为`s6-00343/00387/00426/00433/00451/00476/00530/00592/00593/00406`；前九TRAIN、最后一DEV。

因此该输入版本差异在**当前准入窗口集合中未造成跨段监督**，不能把它误写成已发现实际污染，也不能假称原始pack身份相等或已修复。未来远端重建需显式声明版本与边界合同、发布完整tuple差异；若出现真正跨段/监督错位则立即退回准入并禁止训练。[287条全量独立边界影响审计](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/hot3d-all-native-segment-impact.json)。

## 8. 一条E2E smoke，扩展为九源各一窗（实测）

完整索引adapter→真实collate→真实101,353,749参数PTv3/SharedWAMV6→全部启用监督loss，物理GPU4、严格FP32、batch1；每源一次forward，**所有预测和loss有限，且存在有效监督分母**。loss：s0=0.71905, s1=13.06282, s2=0.99929, s4=4.30355, s5=0.76948, s6=1.55155, s8=9.38570, s9=3.98759, s10=1.09657。耗时16.48s，CUDA峰值约439.46MiB，进程已退出；使用seed随机初始化，仅验证合同，不报告质量改进。

**0 optimizer、0 backward、0 updates、0 checkpoint**；没有以train.py的smoke选项偷跑梯度。生产recipe的拒绝也同场记录，故不能把此结果称为已经完成全量生产launcher/DDP/恢复验收。[逐源loss/监督分母/环境收据](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/smoke.json)。

## 尚未关闭的门与预计修复量（全部为规划估计）

| 优先级/缺口 | 具体要补 | 时间估计与边界 |
|---|---|---|
| Blocker：生产全量recipe/reader接入与身份 | 把稳定sequence_id、完整索引、采样规则、event/unknown policy、TRAIN-only stats和config绑定；保留生产准入；用相同入口复跑无梯度forward | 约2–4工程小时；不含完整内容去重和数据落地，未执行 |
| Blocker：NAS同一数据版本落地 | 方案A复制本地冻结成品并SHA/tuple验收；方案B在NAS原位按已对账输入重建并全量验收；原pack差异必须显式解决，不能凭文件齐全放行 | A按当前小探针约5–7天并可能更长；B不能由32条probe承诺全量时长，先做分源ETA与完整身份对账 |
| 未过：九源DEV/source8 | 恢复官方DEV母序列，母序列隔离、内容与新manifest重验；或明确只做九TRAIN/八DEV受限结论，不能宣称九源DEV已过 | 若原始DEV导出已在盘，估30–90min；缺原始数据则先定位，不虚报采集工期 |
| 旧协议未完：最终内容去重/catalog | 保留r4/r6和STOP；在有预算的新NAS attempt完成代表账本、前后覆盖及新catalog | 先分源小样估计再给工期；原25.317GiB是上界预算，非实测剩余 |
| 仅goal对照阻塞 | 新96条单独冻结合法goal-only配对与独立DEV，禁止把contact派生事件冒充意图证书 | 配方工程估1–2h；若需official DEV资产新采，则另计且尚无实测ETA |

**当前只修上述数据门，不排队、不启动远端4卡训练。** 8格补采不是无goal基础拟合的默认前置动作，事件证书也不靠改标签补造。所有正式效果阈值/在线面板仍NOT_RUN。

环境：本地`/home2/wyy/miniconda3/envs/graspenv/bin/python`，Python3.8.20、NumPy1.24.4、Torch2.4.1+cu121/CUDA12.1；[Conda history](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/conda-history.yml)。本轮17项定向合同测试通过；一次最初测试命令引用不存在文件、0收集，纠正后17 passed，不计前次为通过。`tools/verify.py --changed`未通过：其全局扫描命中其他Task四张20261011历史卡的30个既有失效产物链接，未改动这些材料；本轮四份变更文档的链接已单独检查通过，不冒称全仓校验PASS。

保留的失败/停止attempt：r1样本内容检查后写收据遇相对路径错误；首次goal把canonical instruction ID误当文件字节SHA而被断言阻止，按schema修复后通过；r2逐窗重复验证同一大JSON耗时，主动中断并改成每rollout验签一次、所有goal编码批量核对、192窗实际消费的r3。**均无训练，无数据改写，原日志保留。**

完整机器收据：[可提交JSON](full-data-gates-20261023.json)；[本地完整汇总](../../../../../outputs/ptv3-shared-wam-v6/data-gates-20261023-r2/gate-summary.json)包含证据文件SHA。实际源码在`926d9f35`基础、`188bebd2`独立复核、`288c8dee`全量HOT3D审计中。复现时保持旧STOP，不恢复转换：

```bash
# 对每一stage用全新输出目录，不能覆盖已有audit receipt；均不调用训练入口。
TMPDIR=$PWD/tmp/full-data-gates-20261023 OPENBLAS_NUM_THREADS=1 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/verify_full_data_gates.py \
 --stage full --output outputs/ptv3-shared-wam-v6/<new-audit-run>
# 其他CPU stages: sample, hashes, geometry, goal；smoke额外CUDA_VISIBLE_DEVICES=4，先查空闲。
```
