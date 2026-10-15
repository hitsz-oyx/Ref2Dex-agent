# 原生七源清洗与TRAIN统计

这是用户“先做数据处理、持续推进”授权下的工程处理记录；不构成模型收益、数据质量无误或正式训练准入结论。
此前只读审查中的问题已按下面的实际处理结果更新；历史审查不重写。

## 已完成的快照

完整清洗版入口：[`native3d-clean-mixed-20261015-r1/manifest.json`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/manifest.json)。
保留37,591段、2,638份使用中的几何文件，H4/K24、米制、输出30Hz、stride8。
适配当前v2 encoder的派生入口为`native3d-clean-encoder-20261015-r1`，索引和对应TRAIN统计均已完成，见后节。

| source | TRAIN窗口 | 时钟核实DEV窗口 | 额外待核实时钟DEV |
|---|---:|---:|---:|
| 4 UniDexGrasp | 264,480 | 66,080 | 0 |
| 5 TACO | 30,568 | 7,899 | 0 |
| 6 HOT3D | 71,070 | 16,952 | 0 |
| 7 HO-Cap | 0 | 0 | 8,933 |
| 8 DexYCB | 351 | 0 | 0 |
| 9 H2O | 5,691 | 1,180 | 0 |
| 10 HOI4D | 55,547 | 13,841 | 0 |
| 合计 | **427,707** | **105,952** | **8,933** |

这里是新增native3D工程语料，未并入原有OakInk2/GRAB/ARCTIC三源训练集。窗口数不是独立动作数，
也不是最终source采样比例。HO-Cap只保留DEV，默认reader不包含它。DexYCB当前只有官方s1 TRAIN的subject10，
不制造该来源的DEV。

## 实际清洗

- CAD非零法线单位化，保持方向与固定32个query身份。六源共10,751个点需要重缩放，涉及21份几何；
  6份几何中的8个零/极小法线surface点不在query内，因此只移除这些非query点，不猜测法线方向。
- `TACO/187`的零法线影响query，整份几何及涉及的9段序列隔离；HOT3D另有2段没有合格窗口。
  原始数据、失败记录及被隔离文件均保留。
- 时间步不压缩或删除。原始帧号跳变、时间戳重复/倒退、时间间隙形成窗口边界；HOT3D结合原生质量位
  与16ms对齐阈值过滤。全量读到55,361个native质量位为false的帧、19个重复/倒退边和19个残差超限帧；
  它们可能重叠，不能相加当作独立失败帧。旧pilot的2ms规则没有冒充全量converter规则。
- target在4帧历史与24帧未来全部有效；至少一侧手的11点历史完整，该侧未来有效点比例至少90%。
  另一侧手缺失不会让单手样本全部失效；缺失mask保留，不生成假静止或虚构第二只手。
- UniDex compact观测本已30Hz，逐文件核对其与60Hz `action_time[::2]`一致，不再二次降采样。
- H2O独立TEST的220个文件有明确manifest归属；与524个TRAIN/DEV片段对账为744，没有缺失或无主文件。
  TEST没有进入窗口索引和统计。
- 源manifest快照、源行号、原始episode/sequence、profile、split/group、目标slot和文件size/mtime都保留。
  reader在访问缓存时仍检查payload和窗口索引漂移；旧reader不能绕开清洗索引。

六源实际运行用时481.96秒，代码`9840787c`；完整统计和隔离原因为
[`cleaning-summary.json`](../../../../../outputs/3d-data-preparation/native3d-clean-20261015-r1/cleaning-summary.json)、
[`geometry-ledger.json`](../../../../../outputs/3d-data-preparation/native3d-clean-20261015-r1/geometry-ledger.json)、
[`sequence-ledger.jsonl`](../../../../../outputs/3d-data-preparation/native3d-clean-20261015-r1/sequence-ledger.jsonl)。
summary的`source_counts.rejected_windows`汇总保留序列；若需要计入无合格窗口的序列，须读取完整sequence ledger。

## HOI4D时钟与插值

原1,326段中，1,251段具有300个连续native frame ID、20秒action duration，与作者15Hz时钟一致；
69段duration冲突、6段缺duration，继续隔离。证据见[时钟笔记](2026-10-15-native-clock-cleaning-evidence.md)。

每段保留300个原观测，在相邻有效观测间加入299个半步：点/平移线性插值，旋转使用短弧SLERP中点。
任一邻帧缺标注就不给该点/物体生成有效中点，无外推；最终599帧结束于299/15秒。
`observed_frame`、`interpolated_frame`、native ID与半步位置同时保留；标注有效性另由mask表达。
合计375,300个原观测和374,049个网格插值帧，不能把后者写成新增真实观测。

转换16.18秒、后续窗口清洗17.06秒，代码`cb9d176b`；
[`转换记录`](../../../../../outputs/3d-data-preparation/hoi4d-resampled30-20261015-r1/progress.json)、
[`75段隔离清单`](../../../../../outputs/3d-data-preparation/hoi4d-resampled30-20261015-r1/rejected-sequences.json)、
[`窗口清洗记录`](../../../../../outputs/3d-data-preparation/hoi4d-clean-20261015-r1/cleaning-summary.json)。

## 读取、标签和混合batch检查

- 清洗、索引、插值、合并、统计、encoder view与原有Task合同测试共35项通过；`tools/verify.py --changed`通过。
- 对全部2,638份保留几何核对单位法线与原query身份。
- 按source/split均匀抽96个窗口，独立从世界系物体pose复算未来24步端点，最大误差
  `2.8128766e-8 m`；模型输入有限、全target horizon有效，provenance跨collate保留。
- 七源各一窗的正序/反序混合batch通过；source/sample/profile、anchor、active objects与HOI4D
  历史2/4、未来12/24真实观测标记保留。该batch特意包含DEV/时钟待核实source7，仅用于接口检查。
- 额外逐一检查全部542,592个all TRAIN/DEV窗口的H4-active scene点数，最大3,072，超过默认4,096 cap的为0。
  这补足抽样检查不能排除截断风险的边界；没有据此删除任何窗口。

记录：[`reader-audit.json`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/reader-audit.json)、
[`mixed-reader-audit.json`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/mixed-reader-audit.json)、
[`capacity-audit.json`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/capacity-audit.json)、
[`静态几何审查图`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/cleaned-geometry-review.png)。
初次临时混合脚本缺少Shared-WAM的Python路径而失败，修正为可复用audit入口后通过；未改动数据或模型。

## TRAIN统计处理

`accumulate_clean_moments.py`只读取冻结`verified_train`索引的427,707个窗口，按来源存储hand/scene各24×XYZ
的有效点count、mean、M2，不读取DEV payload或TEST。
初轮代码`72580d23`读取了全部CAD，其中542份没有TRAIN窗口引用，未参与统计但超出约定读取范围；
已在6,656段的安全边界以STOP终止并保留r1记录，修复为只加载TRAIN实际使用的2,096份CAD。
重跑`native3d-train-moments-20261015-r2`代码为`38a31e31`，CPU4workers、30分钟/100MB边界，实际512.41秒完成。
完整结果为[`source-moments.json`](../../../../../outputs/3d-data-preparation/native3d-train-moments-20261015-r2/source-moments.json)。
向量化标签已与实际`pose_mesh_sample`在旋转anchor、部分手缺失和物体未来缺失情况下逐项对照；
专门以不存在的DEV payload和DEV-only CAD验证统计仍能完成，伪造TRAIN索引包含DEV行会失败。
独立复核另从source4/5/6/8/9/10各取一个真实窗口，计数与实际reader全部相同，均值最大差2.42e-8m。

以后指定source概率后，`mix_moments`先将每源累计量除以该源窗口数，再按source概率合成共同统计。
这对应source→均匀window抽样下的global masked-point mean；有效手点/对象多的窗口贡献仍较大，
不是每window严格等权。没有提前选择source权重、std floor或修改现有main3 normalizer。
若以后改成source内部motion/episode分层抽样，须匹配其实际窗口测度重新组合或处理统计，不能直接沿用均匀window假设。
HOI4D的有效插值标签包含在其30Hz网格统计中，原观测与插值标记另保留供后续loss策略选择。

同一遍处理还核对现有v2 encoder配置的当前观测workspace，只生成不兼容窗口清单，不自动删窗。
六个TRAIN来源每个预测步均有hand/scene有效观测，未出现空统计horizon。检查发现41个source4 TRAIN窗口超出当前encoder输入域，
其余TRAIN来源为0；后续单独生成兼容版本，不重写完整清洗版。

## 当前encoder兼容版本

入口：[`native3d-clean-encoder-20261015-r1/manifest.json`](../../../../../outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/manifest.json)，
对应[`source-moments.json`](../../../../../outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/source-moments.json)。
保留**427,666 TRAIN、105,946 verified DEV**；HO-Cap另8,933 nominal DEV仍只在all索引内。
37,591段都有剩余窗口，未改原始序列、坐标、几何或完整parent索引。

v2配置为`voxel_origin_m=[-2,-2,-2]`、`voxel_m=.01`，合法grid为0..65534，
即各轴坐标约在[-2,653.35)米。当前问题出在负方向下界，不能把它解释成对称±2米物理工作空间。
TRAIN的41个窗口涉及12条UniDex执行；CAD仍为厘米级，而原记录物体已离开手附近数米，最远手物距离约13米。
这支持“物体飞离手附近”的工程描述，未证明仿真数值错误；
[`逐条诊断`](../../../../../outputs/3d-data-preparation/native3d-clean-mixed-20261015-r1/workspace-outlier-details.json)保留实际范围。
该兼容性过滤也不等于物理合理性、成功演示或距离阈值过滤，正方向远离样本仍可能合法。

DEV的全部7,562个payload独立核对当前输入后另有6个越界窗口。派生索引排除TRAIN41/DEV6，
原row/tick身份保持；完整排除清单为[`excluded-parent-windows.npz`](../../../../../outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/excluded-parent-windows.npz)。
判定只用当前scene surface及H4-present手点，不用未来、action、reward或成功标签。
DEV只用于自己的输入域判定，没有进入归一化拟合。

TRAIN充分统计用完整版本减去41个排除窗口的贡献，与新索引同步；每个horizon恰好减去41×11个hand点与41×32个scene点，
其他五个来源的数值统计逐项不变。解析减法与直接重算在已知答案测试中一致，future标签改变不影响encoder准入的测试通过。
代码`bf5f9a89`，90.53秒完成；最终全2,638几何、96窗独立端点复算及七源正反混合batch再次通过：
[`reader-audit`](../../../../../outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/reader-audit.json)、
[`mixed-reader-audit`](../../../../../outputs/3d-data-preparation/native3d-clean-encoder-20261015-r1/mixed-reader-audit.json)。

本轮使用CPU4workers，原因是逐文件NPZ解码、mask/时钟处理、小矩阵与充分统计，无神经网络推理或MANO/FK重建；
既有GPU训练保持运行。最终仍为工程数据，`training_allowed=false`。

## 剩余边界

原始与所有清洗manifest保持`training_allowed=false`。reader的`engineering=True`是显式工程访问，
不是翻转准入。后续混合训练需要选定采样recipe后合成共同TRAIN标准化参数，并接入production adapter；
当前v2 normalizer明确只接受main3来源。input-domain已提供显式兼容view。

HOT3D/HO-Cap的first-metadata-object仍只是工程anchor，TACO是原生tool slot，不自动等于任务操纵目标；
不同来源手部根点和机器人形态保留独立profile。当前split继承source group，未声称全七源object-disjoint，
也未做语义近重复去重或静态/运动重新配比。以上工程检查不证明混合训练有效，不启动大规模预训练。
