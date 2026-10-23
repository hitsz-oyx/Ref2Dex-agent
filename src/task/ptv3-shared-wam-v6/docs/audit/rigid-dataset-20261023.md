# 刚体数据集 P0：准入、认证、去重与覆盖（2026-10-23）

| 观测（同一固定面板、同一训练预算） | 含义 | 决定 |
|---|---|---|
| 新增合法覆盖 → 预测误差下降，包括接触窗口与前8帧 | 数据是瓶颈的一部分 | 继续按覆盖表补数据 |
| 新增覆盖 → 预测不动 | 瓶颈在条件/优化/标签，不在数据量 | 停止买覆盖，转向条件与标签审计 |
| 准入修复后合法增量约0 | 当前合同下本地直接可用增量近上限 | 将“是否需要新采集”交回机主，并给成本估算 |

**实测落在第三行：legacy 0 admitted / 1,564 rejected；原6,000窗口重检全部准入，
部署合法goal认证仍0/6,000。保守内容去重6,000→5,999，要求的108个覆盖组合有61个为0。**
没有新数据训练；预测误差变化、接触/first8增益均为 `NOT_RUN`，不能填0或宣称改善。

这是准入规则与绑定路径的工程修复交付，**尚未打通Inspire legacy到v6监督的转换**。
本轮没有实现/认证新的Inspire→joint21语义适配器；缺失标签仍明确拒绝。合法增量0
只针对这6个view的当前合同，不证明1,335条原始母序列的信息已耗尽，不证明只能新采集。
按机主停止条件交回决策，不自行扩源、扩大转换或换模型。

实验：`P-20261023-rigid-dataset-admission`；run：`rigid-dataset-20261023-r1`；
执行代码：`47688afc9a98424000493dab6678df2c0d29565f`，branch `main`。
[机器摘要](rigid-dataset-20261023-summary.json)；
[实验卡](../experiments/probes/P-20261023-rigid-dataset-admission.md)。

## D1：四类阻断的机制与真实重判

旧pilot给sample_window传空占位数组，只捕获首个KeyError；并未逐窗读真实payload。
新工具读取真实数组、元数据、缓存身份、时钟与全H4/K24切片；累计所有可判定理由，
不因第一个缺项崩溃。所有规则不以物体身份设置阈值或选样。固定原先每根TRAIN字典序
前三条：405+227+150=782窗口/根，两根共1,564view窗口，仍只有3条parent。

| 阻断 | 机制与通过条件 | 本轮真实结果 |
|---|---|---|
| 21点语义 | 显式JOINT_NAMES顺序、part语义与正确shape；与实际sample_window合同一致。尺寸相同不能代替语义映射 | 1,564缺joint21及其有效性；未用人类母序列、插值或任意FK点替代 |
| validity | H4至少一个joint连续可观测；t−3…t+24物体pose显式valid且有限SE3；帧边界、mask类型严格检查 | 数值SE3/时钟可核验，annotation validity仍缺1,564；没有全True补齐 |
| contact/part/event | 固定v5 proximity语义、16part范围、显式valid/unknown、hand/geometry/contact/event身份链一致 | 1,564缺对应sidecar；2cm candidate mask不能冒充contact或event |
| surface registry | 2048surface/normals+32query，canonical坐标/米/法线/来源绑定可核验 | coupled的782窗口具备可绑定registry；cm的782因跨view运动不一致仍拒绝绑定 |

实际新生成的是小型独立sidecar：全帧逆物体位姿检验刚体点池后，固定4096池的偶数
point ID选2048点，再固定每64点取query32。写出数值SE3 mask和原位姿；数值mask
名为numerical_pose_valid，不能当object_pose_valid。cm候选registry明确标为UNBOUND。
冻结TRAIN normalizer不重估，旧生产reader/config不放行，新manifest training_allowed=false。

**额外实测缺陷：**cm的row.path失效，但明确声明的view_manifest/q/wrist绝对路径有效，
现已机制化恢复读取。cm和coupled同parent/帧数/时钟不能证明同一运动；
`coupled_pose @ cm_wrist_object_t` 与 `cm_wrist_world` 逐帧只有首帧一致，三条序列
最大矩阵元素差0.5900/0.6639/0.9259（含无量纲旋转元素），平移分量最大差约
0.243/0.533/0.449m。保留各自位姿，不以parent名替cm借用coupled证据。

| root | parent | 候选 | admitted | rejected |
|---|---|---:|---:|---:|
| cm_decoder_v2 | s1/airplane_lift | 405 | 0 | 405 |
| cm_decoder_v2 | s1/alarmclock_see_1 | 227 | 0 | 227 |
| cm_decoder_v2 | s1/apple_pass_1 | 150 | 0 | 150 |
| coupled | s1/airplane_lift | 405 | 0 | 405 |
| coupled | s1/alarmclock_see_1 | 227 | 0 | 227 |
| coupled | s1/apple_pass_1 | 150 | 0 | 150 |

拒绝理由为**重叠计数，不能相加作分母**；完整互斥reason_combinations见JSON。

| reason code | 窗口数 |
|---|---:|
| `contact_part_valid_missing` | 1564 |
| `contact_semantics_unbound` | 1564 |
| `event_contact_binding_unbound` | 1564 |
| `event_missing` | 1564 |
| `hand21_missing` | 1564 |
| `hand21_semantics_unbound` | 1564 |
| `hand_part_semantics_unbound` | 1564 |
| `hand_surface_missing_or_invalid` | 1564 |
| `joint_valid_missing` | 1564 |
| `object_validity_missing` | 1564 |
| `peer_motion_binding_mismatch` | 782 |
| `surface_registry_unbound` | 782 |

现有6,000条则全部通过相同谓词及真实hand/contact/geometry/event元数据身份核验，
证明规则有合法通过路径。有效unknown维持原mask，不伪造负接触。回归中的合法合同
实际进入sample_window；语义置换、代理标签、非法part、坏绑定和越界均拒绝。

## D2：每窗来源台账，缺证保持不可认证

沿原6,000有序tuple精确join既有brief35认证，并重新核对1,064母序列pose文件hash，
未重做已完成来源调查。每条写source_role、available_before_reset、label来源hash/
mtime和缺口。未知独立计划source_role=UNKNOWN、available_before_reset=null；未知
reset/来源可用时间及独立goal来源hash保持null。当前标签明确是recorded_target_rollout，
旧goal使用t+24自身未来；该事实与“是否另有独立计划但缺证”分开记录。
mtime仅说明文件时间，不能证明reset前可得。0认证/6,000不可认证，合法goal索引为空。

五类缺口各覆盖6,000窗：派生来源绑定、显式指令角色、目标执行/reset绑定、reset前
冻结证据、独立于目标rollout的来源。三条Shadow/30窗的native goal候选沿用既有缺证
结论，不能拿另一执行的声明或合成机制正例为旧人类窗口发证。全部现有窗口均有台账，
包含后续被去重剔除的那条；去重后仍是0/5,999具备部署goal认证。

## D3：内容近重复与时间重叠分开

预先固定的保守判据：PCA几何extent各维比值≤1.05；完整28帧H4/K24在当前物体坐标
锚定后的所有joint21/query32坐标差≤2mm；各帧物体旋转差≤1度；mask和phase完全
一致；每帧每part的contact≥.25占2048query比例差≤.02。比较跨序列内容，不用物体
名称/source身份筛选。KDtree只作必要条件索引，最终逐对严格比较；按旧冻结ordinal
保留最早代表，不传递合并。不同canonical CAD朝向/采样可能漏检，是保守上界限制。

- 内容去重：6,000→5,999，删除1（0.0167%），跨序列删除0；没有追调阈值。
- 删除ordinal52（source0、s0-00451、slot0、tick203），保留ordinal51/tick195；
  几何/时间/接触条件均过。原始数组独立复算：最大hand坐标差1.811mm、query差0.354mm、
  物体旋转差0.179度。
- 影响：approach1,200→1,199，source0的96→95；close1,200/contact1,800/hold1,740/
  unknown60不变。1,064序列、9source、61(source,object)实例均保留。
- 原61.2333%是“同序列/slot内tick距离<24”的时间重叠率，去重后61.2269%；这不是
  本规则的内容重复率。**本轮没有解决大多数时间相关性**，也不把59.99百窗叫作59.99百
  独立样本。固定预算对照须仍以原面板及母序列聚类统计，不因去重改变统计单位。

新索引是独立候选产物，旧6k索引/配方/模型未覆盖、未自动接入训练。

## D4：phase × 数值几何类别 × source（去重后）

几何类别由surface PCA包围extent的s1≥s2≥s3决定：flat=s3/s2≤.2；否则
elongated=s2/s1≤.4；其余compact。都是物体身份无关的形状类，不能解释成杯/铰链
等功能类别；不引入非刚体分类。phase沿冻结proximity event合同，非实测force接触。
**粗体0就是缺口。108个要求组合中61个为0，前后相同；其中47个非零组合并不代表分布均衡。**

| phase / geometry | s0 | s1 | s2 | s4 | s5 | s6 | s8 | s9 | s10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| approach / flat | 7 | 53 | 3 | **0** | **0** | **0** | **0** | **0** | **0** |
| approach / elongated | **0** | 131 | **0** | **0** | **0** | 15 | **0** | **0** | **0** |
| approach / compact | 54 | 885 | 2 | **0** | **0** | 49 | **0** | **0** | **0** |
| close / flat | 1 | 53 | 3 | **0** | **0** | **0** | **0** | **0** | **0** |
| close / elongated | **0** | 135 | 1 | 5 | **0** | 1 | **0** | **0** | **0** |
| close / compact | 7 | 984 | 1 | **0** | 4 | **0** | 5 | **0** | **0** |
| contact / flat | 1 | 70 | 6 | **0** | **0** | **0** | **0** | **0** | **0** |
| contact / elongated | **0** | 187 | 1 | **0** | 1 | **0** | **0** | 1 | **0** |
| contact / compact | 6 | 1518 | 3 | **0** | 4 | 1 | **0** | 1 | **0** |
| hold / flat | 7 | 83 | 20 | **0** | **0** | **0** | **0** | **0** | **0** |
| hold / elongated | **0** | 187 | 19 | 22 | 5 | **0** | **0** | 2 | **0** |
| hold / compact | 9 | 1338 | 37 | **0** | 4 | **0** | 7 | **0** | **0** |

另有unknown phase60窗：s0 compact3、s1 flat1+compact7、s4 elongated3、s10 compact46。
这些不强塞入四phase；几何unknown0。完整前后表含unknown维度，见coverage-before/after.csv。

## D5：可比较性与停止决定

原训练身份保留：condition_recipe `36872da2…`、manifest `5bda6a99…`、ordered tuple
`d18ec57e…`、6,000窗、seed170。本轮实测重新读到的r7 manifest SHA前缀也是5bda6a99。
原512锚点 `94d9287bb6c650cc80e4345515e0589472786bd749981a0b25ea7475118ebc37`
及完整anchor_rows单独保存为fixed-panel.json；不从新5,999集合重新选评估例。

合法增量0且goal认证0，不启动匹配训练；同预算预测改善无实测值。今后若有合法新增，
固定该面板、同模型/优化/更新数/采样预算/seed及部署条件，分别报告overall、contact和
first8；新条件若与旧训练身份不同，不能把差异仅归因数据。训练前还须来源/缓存合同
与GPU{4,5,6}显存/进程/存储preflight；7不纳入本轮训练池。

**交回机主的决定：“是否需要新采集？”**当前最具体缺口是有可信21点/part语义、
显式观测validity、正确view几何绑定，以及reset前任务指令/reset/clock证据的刚体数据。
不另起扩源任务；也可由机主选择对本地native数据做明确的语义标注/适配资格化。

下面是**自报的规划假设，不是实测工时/吞吐/报价，也不是启动授权**：

| 选项 | 可审阅的最小预算/成本 | 能改变什么决定 |
|---|---|---|
| 现有仿真来源新采96个刚体rollout，每条10s@30Hz，按当前source的phase/几何缺口配额 | 28,800ticks；按16–32KiB/tick约0.44–0.88GiB，含registry/日志给2GiB硬上限；接入来源和语义审计估1–2人日，采集墙钟先封顶2h，吞吐待GPU preflight后实测 | 能否获得真正可认证增量；只能补该仿真source，不能冒充其余8源覆盖 |
| 本地native语义/绑定资格化后重判原1,564 | 工程初估2–5人日；先资格化3parent，不做96.5GiB全量转换；旧goal reset证据仍可能无法补回 | 能否用显式适配而非新采集挽回部分数据；没有真实reset来源仍不能认证旧goal |

大规模完整新采集的可靠成本需由上述pilot实测后推算；不能从6个view的零准入推成
所有本地原数据都必须重采。既有96.5GiB只是历史扩转换估算，未当作本轮精确需求。

## D6：空间、环境与验证

执行前精确序列化需求 **10,963,775 bytes / 48文件**，按块取整文件空间
11,096,064 bytes，另预留1,048,576 bytes元数据；写前余量
273,116,987,392 bytes，预算后余量273,104,842,752 bytes，远高于20GiB floor。
实际序列化字节数逐文件求和一致。目录/日志块的实际开销是文件系统相关量，没有将预留量
冒充精确元数据占用。计划预检与执行相差7bytes，来自mode标签序列化长度。

执行graspenv Python3.8.20 / NumPy1.24.4 / SciPy1.10.1，93.315s，数据审计CPU/IO，
0神经训练/推理、0物理controls、0checkpoint；已有outputs删除数0。所有输出在新run目录。

验证：新准入与相关合同19 passed；逐窗数/有序来源join/新索引/覆盖CSV独立重算PASS，
删除对的原始数组度量重算PASS。全Task回归242 passed / 1既有fixture失败：
`test_missing_scheduler_binding_fails_before_reading_any_labels` 用object.__new__构造对象，
未设goal_input_policy，报AttributeError（本轮未改该reader/fixture，既有STATE也记录此项）。
不将全库回归称为通过。

`tools/verify.py --changed`已执行，失败项为另一Task
`cm-pointflow-effect-pretrain`四张历史卡的30个既有outputs失效链接；本轮文档链接与
实验卡schema无新增错误。未改非刚体路线来消除这些失败。

并发边界：本轮执行/身份冻结于47688afc。独立会话随后提交eb97e50f的全量刚体准备
工具和独立实验卡；本报告未验收其运行，不用该路线扩大这里1,564候选的结论分母，
也不将本轮零准入解释为该全量路线必须停止。

资源操作偏差：初次全Task回归漏设CUDA_VISIBLE_DEVICES，既有小型CUDA attention梯度
测试在默认cuda:0执行，进程已退出；未提前按本轮GPU池隔离，是本次检查遗漏。数据审计
本身无GPU调用；后续19项定向检查及最终execute显式CUDA_VISIBLE_DEVICES=''。此事不计
作训练授权或新训练数据，也不把审计的GPU0秒外推为整轮所有测试都未触GPU。

复现工具：[build_rigid_dataset.py](../../tools/audit/build_rigid_dataset.py)，
谓词：[rigid_dataset.py](../../src/ptv3_shared_wam_v6/rigid_dataset.py)，
回归：[test_rigid_dataset.py](../../tests/test_rigid_dataset.py)。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/build_rigid_dataset.py \
 --output outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r2 --plan
# 审阅精确预检后，同命令--execute；已存在目录会拒绝覆盖。
```

逐窗和数组证据（保存在outputs，不复制进Git）：

- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/legacy-window-verdicts.jsonl`：1,564个真实候选的全部理由。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/legacy-bindings/`：小型registry、位姿、数值validity与来源绑定；cm候选明确未绑定。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/curated-window-verdicts.jsonl`：6,000个现有窗口的相同准入复核。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/window-provenance.jsonl`：6,000条来源角色/时间/哈希/缺证记录。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/dedup-pairs.json`：删除项到实际保留代表。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/deduplicated-train-tuples.npy`：5,999条独立候选索引。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/deduplicated-ordinals.npy`：相对原6k的有序保留位置。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/legal-goal-tuples.npy`：0条可认证合法goal。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/coverage-before.csv`：去重前180格（含unknown）。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/coverage-after.csv`：去重后180格（含unknown）。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/fixed-panel.json`：原512锚点及身份。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/report.json`：实测运行与汇总。
- `outputs/ptv3-shared-wam-v6/rigid-dataset-20261023-r1/storage-preflight.json`：写前精确字节和余量。
