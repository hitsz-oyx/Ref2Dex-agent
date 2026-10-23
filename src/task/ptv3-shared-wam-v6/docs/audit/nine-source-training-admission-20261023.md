# 九源转换对账与训练准入（2026-10-23）

**结论：九源 TRAIN 已齐；目前不能放行九源大训练，A 继续无 goal/no-event。**
转换于 **2026-10-23T11:57:25+08:00** 完成，固定转换剩余 **0 B**，ETA 已由实测完成时间替代。
独立全量 census 于 **2026-10-23T04:04:57.088191+00:00** 完成，39,527/39,527 条母序列，
TRAIN **9,341,050** / DEV **1,856,282** 窗口；
全部通过 native clock/segment 的 H4/K24 连续性检查，时钟新增拒绝0。
DEV 只有8源，**source8 DexYCB 的 DEV=0**，不能称九源 DEV 齐全。
108个TRAIN phase×几何×source格中，零格 **61→8**，减少53格（86.89%），原非零格无丢失。

转换后快照可用空间 **70.252 GiB**；这不是训练空间已预留的证明。
全窗口[manifest](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/manifest.json)、
[全矩阵 CSV](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/phase-geometry-source.csv)、
[覆盖热图](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-report-support/coverage.png)、
[独立检查](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-report-support/independent-check.json)。
manifest SHA256：`4e40b4ec0bd58d390f8d5cee6047530d1832c3d5acb6080232251236d934c4c3`；实际 census 代码提交 `e1c164cea08bb968195e8d5508f958251572f63e`。

### 九源窗口与相对6k净增量（实测）

“可用候选”指该母序列/目标slot的 dense H4/K24 起点，尚未通过逐窗几何/validity准入；
“准入”仍含重叠窗口，不能当作独立样本。九源均有TRAIN，source8缺DEV。

| source | 母序列总数（TRAIN/DEV） | 可用候选 | 准入总数 | TRAIN | DEV | 旧6k TRAIN | TRAIN净增 |
|---|---|---|---|---|---|---|---|
| 0 OakInk2 | 568（498/70） | 6,384,141 | 6,384,141 | 5,412,663 | 971,478 | 96 | 5,412,567 |
| 1 GRAB | 1,133（1,041/92） | 313,434 | 313,434 | 287,955 | 25,479 | 5,632 | 282,323 |
| 2 ARCTIC | 301（267/34） | 420,292 | 420,292 | 370,362 | 49,930 | 96 | 370,266 |
| 4 Shadow replay | 33,055（26,447/6,608） | 2,413,015 | 2,413,015 | 1,930,631 | 482,384 | 30 | 1,930,601 |
| 5 TACO | 2,308（1,820/488） | 299,705 | 299,705 | 238,297 | 61,408 | 18 | 238,279 |
| 6 HOT3D | 287（228/59） | 984,582 | 716,080 | 578,181 | 137,899 | 66 | 578,115 |
| 8 DexYCB | 100（100/0） | 4,582 | 2,953 | 2,953 | 0 | 12 | 2,941 |
| 9 H2O | 524（430/94） | 53,210 | 53,210 | 44,110 | 9,100 | 4 | 44,106 |
| 10 HOI4D | 1,251（1,002/249） | 715,572 | 594,502 | 475,898 | 118,604 | 46 | 475,852 |
| 合计 | 39,527（31,833/7,694） | 11,588,533 | 11,197,332 | 9,341,050 | 1,856,282 | 6,000 | 9,335,050 |

旧6,000行全部能在新TRAIN找到，精确集合新增 **9,335,050** 行，旧行缺失0。
TRAIN母序列31,833条，相对旧6k的1,064条净增30,769条；完整母序列决定39,527，其中新转换38,350、只读复用1,177。
窗口拒绝原因可重叠，独立检查收据保存逐类次数，不与被拒窗口数重复相加。

### 每源 phase×几何 TRAIN 覆盖（实测）

每格依次为 **flat / elongated / compact** 窗口数，unknown phase单列；geometry unknown=0。
完整DEV矩阵见CSV，source8的DEV整列为0。

| source | approach | close | contact | hold | unknown phase |
|---|---|---|---|---|---|
| 0 | 887,017/1,100,703/2,428,525 | 16,359/18,741/46,812 | 17,042/18,500/46,479 | 115,015/341,760/370,980 | 4,730 |
| 1 | 2,879/7,156/52,036 | 1,530/3,817/31,210 | 1,864/4,928/46,237 | 6,425/16,852/110,979 | 2,042 |
| 2 | 13,103/10,302/41,975 | 9,617/3,664/18,647 | 13,286/3,982/23,326 | 83,610/27,758/120,026 | 1,066 |
| 4 | 245/5,783/18,084 | 2,900/63,755/165,526 | 422/24,574/85,898 | 18,043/374,996/1,122,358 | 48,047 |
| 5 | 1,173/4,260/5,536 | 3,772/12,204/14,801 | 4,087/11,524/11,936 | 23,993/72,424/70,612 | 1,975 |
| 6 | 22,813/73,737/368,531 | 1,614/1,718/9,185 | 1,929/1,477/8,135 | 8,223/11,605/54,730 | 14,484 |
| 8 | 2/0/0 | 76/214/1,094 | 0/0/17 | 47/184/1,122 | 197 |
| 9 | 20/6,047/89 | 180/1,374/789 | 216/1,448/862 | 5,047/14,856/12,487 | 695 |
| 10 | 0/6,753/23,571 | 0/5,590/18,441 | 0/5,585/17,846 | 0/28,425/115,089 | 254,598 |

剩余8格：DexYCB source8 的 approach×elongated/compact、contact×flat/elongated；
HOI4D source10 的 flat×approach/close/contact/hold。前者是当前样本/phase缺口，后者是准入TRAIN中
flat几何类缺失；不能用其他source、unknown或新采source4替代。全量几何实例数按(source,object)计
TRAIN为2,218、DEV为688，不是跨source同CAD去重数。

### 超出96.5 GiB的精确对账（实测字节）

旧估算只覆盖选出的6,000条pending sequence，当前是39,526计划全量加1条本地母序列，
不是6,000窗口。把**相同6,000条**按当前写入合同复算后反而减少0.467482 GiB；增加的主要是范围。
固定转换文件由19个NPY和3个65536 B metadata组成；dense tick增加的是窗口索引/准入工作，
没有为每个重叠窗口复制整条轨迹。

| 项目 | 精确字节 / GiB |
|---|---|
| 原6,000条header估算 | 103,613,572,400 B / 96.497659 GiB |
| 相同6,000条当前精确合同 | 103,111,616,953 B / 96.030177 GiB |
| 其余范围新增固定转换文件 | 79,219,577,025 B / 73.778980 GiB |
| 当前全量固定转换合计 | 182,331,193,978 B / 169.809157 GiB |
| receipt逐条求和 | 182,331,193,978 B / 169.809157 GiB |
| hands+contacts+events实际逐文件求和 | 182,331,193,978 B / 169.809157 GiB |
| 重复/partial超额 | 0 B / 0.000000 GiB |
| 固定转换剩余 | 0 B / 0.000000 GiB |

全量固定字节相对旧估算的增加为 **78,717,621,578 B / 73.311498 GiB**，
恒等式：`103,613,572,400 − 501,955,447 + 79,219,577,025 = 182,331,193,978`。
原inventory只有本地可直接转换部分；后续恢复ARCTIC 295条增加4,853,038,583 B、
OakInk2 562条增加49,213,063,217 B、额外GRAB母序列增加2,214,221 B。
这些恢复量已经包括在全量合计，不能再加一遍。v3–v7预算保留了旧的
`conversion_fixed_exact_gib=119.454...`显示值，已与bytes不一致；本报告全部从bytes重算。

run目录扫描快照（不含随后生成的manifest本身与报告/热图）逐文件逻辑字节 **186,161,021,383 B / 173.375962 GiB**，
文件实际blocks **188,702,429,184 B / 175.742832 GiB**；
差额是文件块分配，目录/inode/其他任务不在此精确文件求和内。其他主要项目：

| 目录 | 逻辑字节 | 文件分配blocks字节 |
|---|---|---|
| receipts | 2,590,441,472 | 2,590,527,488 |
| windows | 187,119,384 | 275,361,792 |
| restored-source0-inputs | 660,748,339 | 667,439,104 |
| recovered-native-inputs | 104,494,668 | 107,470,848 |
| readiness-manifest-r1 | 249,636,408 | 249,655,296 |
| batches | 3,778,419 | 5,914,624 |

扫描未发现partial文件或hardlink重复别名；转换固定文件实际总数843,700（38,350×22）。
新转换固定字节与receipt及当前合同三者完全相等，**没有重复写盘导致超估算的证据**。
旧缓存只读复用6,007,523,501 B / 5.594942 GiB，没有复制进新转换目录。
共享文件系统的free变化还含其他运行和元数据；不能把全盘下降全部归因于此次转换。
11:54末段盘点剩余固定文件约6.40GiB，现已全部完成；progress旧的十几秒ETA混入续跑时跳过的receipt，
未用于当前结论。现有失败summary仅写了12,761,987 B，保留原目录，不计其去重成功。

manifest/去重的既有保守上界 **25.317 GiB** 仍是规划上界，
不是还需写这么多的精确预测。另一个活跃`t38-r4`在资源快照已写3.793GiB，
按其32GiB硬帽保守预留剩余28.207GiB；再加新训练16GiB，
三项同时预留后余量只有 **0.720 GiB**，低于30GiB，
因此**当前空间准入HOLD**。可以待阶段完成后按实际重新预算，不能偷偷删历史产物。

### 来源可补性分母（实测；当前文件范围）

| source | 准入窗口 | 已认证 | 当前文件不能补完整证书 | 有native goal候选 | 有动作/verb候选 |
|---|---|---|---|---|---|
| 0 | 6,384,141 | 0 | 6,384,141 / 6,384,141 = 100% | 0 | 0 |
| 1 | 313,434 | 0 | 313,434 / 313,434 = 100% | 0 | 0 |
| 2 | 420,292 | 0 | 420,292 / 420,292 = 100% | 0 | 0 |
| 4 | 2,413,015 | 0 | 2,413,015 / 2,413,015 = 100% | 2,413,015 | 0 |
| 5 | 299,705 | 0 | 299,705 / 299,705 = 100% | 0 | 0 |
| 6 | 716,080 | 0 | 716,080 / 716,080 = 100% | 0 | 0 |
| 8 | 2,953 | 0 | 2,953 / 2,953 = 100% | 0 | 0 |
| 9 | 53,210 | 0 | 53,210 / 53,210 = 100% | 0 | 53,210 |
| 10 | 594,502 | 0 | 594,502 / 594,502 = 100% | 0 | 0 |

全量 **0/11,197,332** 可认证、**100%**缺少可绑定的完整事前来源链；
已检查全部39,527条母序列，发现完整证书字段组合0。
“当前不能补”不等于“已证明永久不可恢复”：外部原始档案待查比例100%，永久丢失比例未知。
Shadow的2,413,015行goal候选、H2O的53,210行语义标注不计认证。

### 当前准入处置

数据九源TRAIN数量门PASS、全量clock门PASS、旧身份门PASS、覆盖增量门PASS；
**全量生产reader/A合同验收、去重/跨split内容证据、保守空间预算、GPU4/5/6资源门尚未全部PASS**，
因此当前大训练 **HOLD**；成功率/新预测质量均NOT_RUN。
既有summary遇官方TEST MANO view没有view_manifest而失败（原日志保留）；修复后独立重放
1,335个legacy mother和633个view，1133个TRAIN/DEV母序列准入、202个official TEST不纳入，
派生view仍0准入。未自动重启失败summary，未把工程异常归因为数据或模型负结果。



## 身份、计数和证据范围

本轮不开训，新增训练 updates=0。A 合同仍是无 goal / no-event；来源缺证不阻断
A 的标签训练资格，但不能据此开启 goal。全量转换与独立审计分开记录；
“转换完成”不等于“已接入生产 reader / 可以开训”。本报告所有实测数取自独立
`readiness-manifest-r1/manifest.json`，下方训练预算和阈值是预注册规划，不是实测收益。

全量 manifest 使用规范化关系：`train-all.npy` / `dev-all.npy` 保存每一行
`[sequence_id, slot, tick]`，同顺序 `*-clock-valid.npy` 给出时钟连续性，
`sequences.jsonl` 将 sequence_id 绑定到原始路径、source、split、hand/contact/event
metadata identity、窗口 shard SHA 和来源字段。每个窗口通过其 sequence_id 继承来源
证据；没有为每行重复写一份相同的大 JSON。manifest 的 files 表保存这些文件的 SHA。
原 padded receipt 不改写，singleton-list split 只在派生结果中与冻结计划核对后规范化。

原 curated 6,000 的 manifest SHA 保持
`5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`，
ordered tuple SHA 保持
`d18ec57ee2de589152131eb2dfab4ac7581ab29275d08756c973dce2c000c3d7`。
固定锚点仍为 512 行 / 124 母序列，逻辑 ID
`94d9287bb6c650cc80e4345515e0589472786bd749981a0b25ea7475118ebc37`，
文件 SHA `440c00c8609a2e9c6276f328f1aa5136a04708294a09b5dab8321044e6caa706`。
逻辑锚点 ID、文件 SHA、代码提交、experiment_id、run_id 不混为一个身份。

窗口净增量是稳定 `(sequence_key, slot, tick)` 集合差，**不等于独立内容净增量**。
全 tick 窗口高度重叠；时间近重复、跨母序列内容近重复以及跨 TRAIN/DEV 内容匹配
由既有 `full-manifest-r1` 去重任务另行审计。未完成前不报告去重后数量或宣称无泄漏。
旧 6k 与全量索引互相独立，不能直接用新 sequence_id 代替旧 sample_id。

## 来源绑定：哪些能补，哪些当前补不出

`source_available_timestamp_ns=None` 首先来自转换器的显式缺省：
`process_one()` 将所有 legacy 的 source_role 写为 UNKNOWN，reset/availability 为 null。
它没有读取任何真实 reset 绑定来签发认证。独立审计进一步逐母序列检查实际 native
payload 的字段，以及 source0/2 的绑定 metadata，区分“writer 没接”与“当前产物无证”。
文件 mtime、采样 clock、pose 文件 SHA、action label 都不能单独补成 reset 前可用时间。

| 类别 / source | 当前保留的信息 | 缺什么、能否补 |
|---|---|---|
| 人类录制：0 OakInk2、1 GRAB、2 ARCTIC | 观测轨迹、有效性、物体身份；0 有源内容 SHA，1/2 有原始数据路径 | 没有可绑定到一次部署 reset 的事前目标 pose/pose rule、角色及同 clock 的可用性证明。当前绑定文件不能补；若找到原始采集指令和对应执行/时间链，可重新审计，不能从未来轨迹倒推 |
| 仿真回放：4 Shadow | `goal`、`reset`、相对 `time` 和状态/动作；goal 为录制的任务参数候选 | 比其余来源更接近可恢复，但快照没有事前 TaskSpec、完整意图 pose/rule、source hash 与 reset 前可用时间绑定。须找回原 producer 的不可变指令/reset 记录再 join；单凭常量 goal 或 reset flag 仍不可认证 |
| 多物体人类录制：5 TACO、6 HOT3D | pose/time/validity；6 另有 native_timestamp_ns | TACO 首工具、HOT3D 首 metadata object 是工程 anchor，不能冒充意图目标；观测纳秒时间也不是指令可用时间。需独立意图/角色标注及原始事前证据，当前不能补 |
| 8 DexYCB | 原生 grasp-object index、观测帧 | 可补目标对象身份，不能由对象身份推出完整目标 pose 或事前可用性；当前不能补完整证书 |
| 9 H2O | native_action_label / native_verb_label | 可用于任务语义候选审计；事后动作标注不自动等于事前 TaskSpec，也没有目标 pose/来源时间绑定，当前不能补完整证书 |
| 10 HOI4D | 原生主要刚体对象、观测/插值标记、动作时钟 | 缺同上事前意图/reset/source 链；不得以末帧 pose 或插值值充当合法 goal，当前不能补 |

这里“当前补不了”限定为**此次实际绑定/恢复的文件**。不能把它扩大成“全世界再也找不回”；
永久不可恢复比例未建立，外部原始档案的可恢复性仍未知。当前产物无证的精确窗口
分母/比例和每源 native goal/动作标注候选数见上方数表及机器收据。

另有独立新采 source4 pilot：96/96 rollout、26,304/26,304 窗口已具备事前绑定，
证据见 [pilot 审计](rigid-source-bound-pilot-20261023.md)。这批数据没有混进本次
39,527 条 legacy census；不能拿它给旧九源签发认证，也不把它算作旧数据补回。

## 硬停止与幂等续跑

转换器原有写前预测 30 GiB floor；新增 `watch_rigid_storage.py` 每秒独立检查实际空间，
覆盖同 run 的转换、完整去重 summary、独立 census。触线 `<30 GiB` 时先 fsync
独立 storage-stop receipt、STOP 及 supervisor terminal marker，再向核实属于该 run 的
同用户独立 session 发 SIGINT；5 秒后仍存活则 SIGSTOP 冻结写入，无自动恢复。
只操作带本 run 精确 argv 的指定工具 session，不干预另一会话的 GPU 训练。
监控有 6h 上限，到期也停止本 run 的写入，不能在监控退出后任其无界运行。
30 GiB 是本轮收紧的边界；其他任务可能继续写盘，不能宣称控制了整个文件系统。

完成序列的原子 receipt、窗口 shard 和批次 receipt 保留。续跑必须先处理 terminal marker
和仍暂停的自有进程，核实空间/输入身份后由显式操作恢复；不得同时启动第二个 writer。
转换命令校验完成 shard SHA 和 metadata identity 后跳过已完成序列。
summary 和本审计必须使用新输出目录，已有尝试保留。没有删除历史产物或覆盖旧 checkpoint。

## 九源大训练准入清单（只登记，不执行）

| 门 | 放行判据 | 当前处置 |
|---|---|---|
| 数据数量与身份 | 全计划 39,526 + append-only local parent 的决定齐；九源 TRAIN 均非零；TRAIN/DEV split、母序列身份和 shard SHA 一致；H4/K24 时钟不能跨 gap/segment；冻结新 manifest SHA | 以本报告实测表逐项核对 |
| 内容与 split | 发布去重/近重复及跨 TRAIN/DEV 匹配收据，明确最终保留集合；若保留重叠窗口，记录其权重/有效曝光，不能按独立样本解释 | 独立 full-manifest 去重尚未完成前 HOLD |
| 实际 reader | 新建全量派生 recipe，绑定 hand/contact/event/geometry/stats；原 6k 不动；真实九源混批、首尾窗口及 DataLoader→collate→model 边界逐位验 A 输入；未来标签干预不得改变条件 | 本轮只做准入清单，不接生产 reader；旧 18-window smoke 不能代替全量 A 合同验收 |
| 第一版配方 | 继承 A 的模型、loss、GT2048 contact、H4/K24/30Hz、stats、初始化、LR/precision、sampler 算法与 seed；只改变窗口池及必要的绑定身份 | no goal/no-event，不接回未来 endpoint/event teacher，不换模型、不同时改采样/预算 |
| 训练预算 | 第一版与实际 6k A 的 2,500 updates / global288（local96×world3）匹配，最多 10,000s（2h46m40s），先到即止；weights-only 同 parent step11000，新 optimizer、seed170 | 是下一轮规划，不是启动授权；不照搬旧已过期 hard_stop_epoch；延长训练另立预算决策 |
| 写盘预算 | 新 run 16 GiB 硬上限，启动时须 `free − 全量 summary 尚需上界 − 其他活跃任务尚需上界 − 16 GiB >=30 GiB`；不足就 HOLD | 见下方精确计数与保守预算；不以瞬时 free 冒充已预留空间 |
| GPU | 仅物理 GPU4/5/6 DDP，启动前确认可用、显存满足并无冲突；不抢占、不中断其他会话、不改派 0–3/7 | 当前其他会话占用4/5/6，须结束并重新 preflight；共享空卡约束按当前 CAMPAIGN/机主预约再次核实 |
| 故障门 | 任一 rank 异常、OOM、非有限 loss/grad、loader/IPC 错误、身份漂移、盘<30 GiB、run>16 GiB、超时/超步均停；保存可用收据，禁止自动重启 | 未来 launcher 必须实际实现，不能只写文档后视为 PASS |

checkpoint 写盘规划（自报预算）：当前实测完整 checkpoint 约 1,217,116,001 B
（1.134 GiB），weights-only init 约 405,716,522 B。按 **1.25 GiB/完整份** 计：
step1/128/500/1000/1500/2000/2500 共7份；可提前停止再保留1份，共最多8份=10 GiB；
init≤0.4 GiB、原子保存临时文件≤1.25 GiB、日志/固定面板/新索引≤3 GiB，合计≤14.65 GiB，
向上封顶16 GiB。不另复制 best/trajectory-best，不额外密集存盘。任何保存计划变化先重算预算。
已有其他训练按其32 GiB上限减当前实测产物保守计剩余，不把该 run 已写文件再扣一遍。

## 固定面板预注册判据

沿用当前 A scorer 的三项主指标，顺序固定为**前8帧手 EPE / 固定接触窗手 EPE /
前8帧物体 EPE**，单位 mm；旧 overall hand EPE 只作辅助，不能换主指标挑好结果。
固定512锚点/124母序列、原 masks/contact>0.25、strict FP32、flow_steps4、flow seed172；
先验证同权重 A 首批与冻结 scorer 一致。每项用原124母序列配对 cluster bootstrap
4,000次、95%CI；不能改成帧级 bootstrap 或重新抽锚点。

对照是**同起点、同 A、同 2,500 updates/global288** 的旧6k与全量新池。
当前 step1223 的 A EPE 10.288421/31.759798/6.504308 mm 仅作已知参照，
不能与新2500步的结果直接比较并归因为扩集。若实际6k对照没有按预算完成，
需要先取得匹配点，不能偷换为历史 B 条件或不同预算。中途面板只用于故障/诊断；
最终判据固定在匹配的2500步或两臂事前固定的共同预算点。

| 最终观测 | 解释与下一步 |
|---|---|
| 三项均相对下降≥5%，且三项配对差 `full−6k` 的95%CI上界均<0；覆盖零格减少且不丢旧格 | “值得”：在固定已暴露 TRAIN 面板、同预算下扩池有预测收益，可进入独立执行验收；不能宣称泛化或成功率提升 |
| 覆盖增加，但三项无一致显著改善或未达5% | 当前模型/条件/采样与这份预算下未证明扩池价值；先审计曝光稀释、有效覆盖、优化/标签，不直接加预算或断言数据无效/无goal已到上限 |
| 只改善一部分，或至少一项显著变差 | 报告混合/回归，保持 HOLD，查该子集和source/phase；不能以均值掩盖回归 |
| 新增主要是时间近重复、零格未减少 | 数量增长尚未建立所需机制增量；先处理采样/缺口，不把更多 tick 当成更多独立演示 |
| 时钟、绑定、reader 或资源门失败 | 工程 BLOCKER，不作为模型或数据价值的科学负证据 |

覆盖价值预注册：108个 `4 phase ×3 PCA几何类 ×9 source` TRAIN 格中，相对原6k
61个零格至少减少 **6格（≥约10%）**，所有原非零格仍非零；unknown单列，不用来填格。
该标准是下一步训练投入的覆盖门，不是每格都已齐的声明；当前覆盖是否命中见实测段。
即使固定面板预测改善，也必须另报成功率：本轮成功率变化=NOT_RUN，不能据EPE归因
执行瓶颈。固定TRAIN锚点不能测量新九源区域的泛化，残余零格是结论适用范围。

## 紧随其后的 goal 独立消融

本轮 A 配方不变。后续仅在可认证绑定非零且训推消费合同通过后，比较
“合法 task-completion goal”与“goal unknown”；两臂同一精确窗口集合/顺序规则、同初始化、
模型、seed、训练步数/global batch/时长/写盘上限，event两臂始终 unknown，唯一变量是
合法 goal 及其 valid mask。对不能认证的行，两臂均保持 unknown，不能从未来pose补条件。
重用同固定面板，并补充认证子集分层报告；若固定旧512行完全无认证目标，则旧面板只能
测间接训练影响，goal直接效应必须在另行冻结、两臂相同的认证面板测量，分母不能混用。

若最终只有上述新采96条source4可认证，最多可先做**source4限定**同数据同预算消融；
不能称作完整九源合法goal消融。若只用认证子集，unknown臂也必须用同一子集。
原始未来端点+B事件的历史诊断结果不作为合法goal正证据或训练配方。

## 复现与检查

转换 experiment 为 `P-20261023-full-rigid-data`，run 为
`full-rigid-20261023-brief47-r1`；实际各次转换代码/环境在 run-manifest*.json。
本次独立 census 的代码提交和运行环境写入其 manifest；graspenv / Python3.8.20，
NumPy版本以 manifest 实测为准。没有神经网络批推理或物理回放，使用CPU/I/O审计。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/audit_nine_source_readiness.py \
 --run outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1 \
 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/<new-audit-attempt>
```

定向测试18项通过：split历史表示及身份漂移拒绝、真实完整manifest serial/spawn路径、
H4/K24时钟跨segment过滤、精确NPY字节、准入向量/标量一致、停止receipt先于信号。
首次测试因未设置 PYTHONPATH 在收集阶段失败；补齐v5/v6路径后18通过，不计首次为通过。
原6k/512身份保留；不修改任何已有模型或训练配方。

`tools/verify.py --changed` 未全通过：仅报告另一Task的4张历史卡共30个既有outputs链接缺失，本轮新增schema/链接无报错；未改无关历史文件或补空产物。
