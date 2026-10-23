# 全量数据准备暂停报告 — brief47 / 2026-10-23

**结果解读表**

| 观测 | 含义 | 决定 |
|---|---|---|
| TRAIN全量准入零格61→8/108；完整内容去重未完成 | 存量转换已完成，不能宣布全量准备验收通过 | 保留已完成数据与全量前去重索引；重训/续训排期仍需补齐去重和读取合同门 |
| TRAIN仍8个零格；DEV 20个零格，DexYCB DEV=0 | 本地准入数据存在来源/几何/相位和split缺口 | 逐格列清，是否补采集/标注/开发集由机主决定 |
| 停止时60.130GiB，硬线30GiB | 本次停止来自新机主brief47b HOLD，未触发空间不足 | STOP和暂停收据保留，automatic_restart=false；不自行越过HOLD |
| 历史goal认证0/11,197,332，独立新分区26,304已认证 | 认证状态分区统计，标签哈希/mtime不充当reset前证明 | 新96条未并入总catalog；不以其证书给历史数据认证 |

## 完成范围与未完成项

“全量”按真实原始payload URI作母序列并集：原冻结九源39,526条 + 原计划外唯一非TEST
legacy母序列1条 = **39,527条准入**。1,335条本地legacy中1,133条准入、202条官方TEST排除；
这202条属于协议排除，不是物理质量非法。原1,064条active TRAIN母序列/1,177条缓存与这些
集合相交，不能重复相加。633条派生view逐条拒绝，缺native21/validity、778/part16或绑定
contact/event；含63条qualitative MANO，未把human parent标签贴给Inspire。

原pending6,000是**序列批次**，已全部完成，不是只准备6,000窗口。总计38,350条新转换、
1,177条只读复用，剩余0；新固定数据精确 **182,331,193,978 B**，与逐receipt/实际文件对账一致。
全量TRAIN **9,341,050**、DEV **1,856,282** 个 `(sequence_id,slot,tick)`；全部native30Hz/
segment边界审计通过。旧TRAIN6,000行全部存在，精确新增9,335,050行；旧DEV为3,356行，
其最终逐项差异及去重后分布未完成，不能用TRAIN新增数替代DEV对账。

逐母序列判定：[九源39,527条](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/sequences.jsonl)；
[legacy1,335条](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/full-manifest-r4/legacy-mother-verdicts.jsonl)、
[派生view633条](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/full-manifest-r4/legacy-view-verdicts.jsonl)。
完整逐源合同、fixed bytes及重复/partial对账见[九源独立报告](nine-source-training-admission-20261023.md)。

**未完成：完整内容去重、全量逐窗认证账本、去重后108格与精确旧TRAIN/DEV差异、
历史池+新96条统一catalog。** r4在20,672/39,527条处被HOLD停止，已处理 **9,418,140**窗，
该前缀删除 **253,211**，还有 **1,779,192**窗未核对内容去重。此删除数只属于已处理前缀；
全量去重后行数为NOT_COMPLETED。r6同样收到STOP，未完成。r4/r6目录的预分配NPY包含未填区，
不作为完整索引使用；完整前去重tuple索引使用下方readiness目录。

## 完整前去重manifest与分布

| source | 母序列 | TRAIN | DEV | 旧6k TRAIN | TRAIN精确新增 |
|---|---|---|---|---|---|
| 0 OakInk2 | 568 | 5412663 | 971478 | 96 | 5412567 |
| 1 GRAB | 1133 | 287955 | 25479 | 5632 | 282323 |
| 2 ARCTIC | 301 | 370362 | 49930 | 96 | 370266 |
| 4 Shadow replay | 33055 | 1930631 | 482384 | 30 | 1930601 |
| 5 TACO | 2308 | 238297 | 61408 | 18 | 238279 |
| 6 HOT3D | 287 | 578181 | 137899 | 66 | 578115 |
| 8 DexYCB | 100 | 2953 | 0 | 12 | 2941 |
| 9 H2O | 524 | 44110 | 9100 | 4 | 44106 |
| 10 HOI4D | 1251 | 475898 | 118604 | 46 | 475852 |

| PCA几何类 | TRAIN | DEV |
|---|---|---|
| flat | 1265498 | 223691 |
| elongated | 2360134 | 510723 |
| compact | 5715418 | 1121868 |

| phase | TRAIN | DEV |
|---|---|---|
| approach | 5080340 | 949248 |
| close | 453630 | 97745 |
| contact | 351600 | 68622 |
| hold | 3127646 | 659796 |
| unknown | 327834 | 80871 |

几何由冻结PCA extent规则分类，不按物体名称；phase来自原proximity协议，不代表力测量或
独立接触意图。source0 dense slot含背景物体，窗口量不等于独立有效交互数。只采用原刚体
SE3/独立slot合同，不新增关节、可变形或关系型目标字段。

## 全量覆盖：4phase × 3几何 × 9source

以下都是完整前去重计数，0格显式保留；unknown另列于原CSV。去重后表尚未完成。

### TRAIN 全量准入前去重覆盖

| phase / 几何 | 0 | 1 | 2 | 4 | 5 | 6 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| approach / flat | 887017 | 2879 | 13103 | 245 | 1173 | 22813 | 2 | 20 | 0 |
| approach / elongated | 1100703 | 7156 | 10302 | 5783 | 4260 | 73737 | 0 | 6047 | 6753 |
| approach / compact | 2428525 | 52036 | 41975 | 18084 | 5536 | 368531 | 0 | 89 | 23571 |
| close / flat | 16359 | 1530 | 9617 | 2900 | 3772 | 1614 | 76 | 180 | 0 |
| close / elongated | 18741 | 3817 | 3664 | 63755 | 12204 | 1718 | 214 | 1374 | 5590 |
| close / compact | 46812 | 31210 | 18647 | 165526 | 14801 | 9185 | 1094 | 789 | 18441 |
| contact / flat | 17042 | 1864 | 13286 | 422 | 4087 | 1929 | 0 | 216 | 0 |
| contact / elongated | 18500 | 4928 | 3982 | 24574 | 11524 | 1477 | 0 | 1448 | 5585 |
| contact / compact | 46479 | 46237 | 23326 | 85898 | 11936 | 8135 | 17 | 862 | 17846 |
| hold / flat | 115015 | 6425 | 83610 | 18043 | 23993 | 8223 | 47 | 5047 | 0 |
| hold / elongated | 341760 | 16852 | 27758 | 374996 | 72424 | 11605 | 184 | 14856 | 28425 |
| hold / compact | 370980 | 110979 | 120026 | 1122358 | 70612 | 54730 | 1122 | 12487 | 115089 |

### DEV 全量准入前去重覆盖

| phase / 几何 | 0 | 1 | 2 | 4 | 5 | 6 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| approach / flat | 154486 | 0 | 2120 | 53 | 217 | 5479 | 0 | 6 | 0 |
| approach / elongated | 235409 | 602 | 2091 | 2500 | 1255 | 19263 | 0 | 1846 | 1616 |
| approach / compact | 420294 | 4935 | 5675 | 5130 | 1281 | 78929 | 0 | 17 | 6044 |
| close / flat | 2684 | 0 | 1587 | 1075 | 926 | 302 | 0 | 55 | 0 |
| close / elongated | 2915 | 394 | 545 | 16426 | 2791 | 511 | 0 | 393 | 1406 |
| close / compact | 6567 | 2944 | 3099 | 39881 | 4625 | 3774 | 0 | 183 | 4662 |
| contact / flat | 2690 | 0 | 2112 | 354 | 907 | 412 | 0 | 59 | 0 |
| contact / elongated | 2911 | 528 | 566 | 6088 | 2462 | 391 | 0 | 319 | 1373 |
| contact / compact | 6680 | 4146 | 3968 | 20882 | 3565 | 3500 | 0 | 165 | 4544 |
| hold / flat | 21142 | 0 | 11581 | 6495 | 5126 | 2342 | 0 | 697 | 0 |
| hold / elongated | 55473 | 1190 | 3215 | 97928 | 16344 | 2669 | 0 | 3166 | 7446 |
| hold / compact | 59483 | 10558 | 13235 | 273544 | 21231 | 16771 | 0 | 2056 | 28104 |


TRAIN的全部0格：

| source | phase | 几何类 |
|---|---|---|
| 10 | approach | flat |
| 8 | approach | elongated |
| 8 | approach | compact |
| 10 | close | flat |
| 8 | contact | flat |
| 10 | contact | flat |
| 8 | contact | elongated |
| 10 | hold | flat |

DEV的全部0格：

| source | phase | 几何类 |
|---|---|---|
| 1 | approach | flat |
| 8 | approach | flat |
| 10 | approach | flat |
| 8 | approach | elongated |
| 8 | approach | compact |
| 1 | close | flat |
| 8 | close | flat |
| 10 | close | flat |
| 8 | close | elongated |
| 8 | close | compact |
| 1 | contact | flat |
| 8 | contact | flat |
| 10 | contact | flat |
| 8 | contact | elongated |
| 8 | contact | compact |
| 1 | hold | flat |
| 8 | hold | flat |
| 10 | hold | flat |
| 8 | hold | elongated |
| 8 | hold | compact |

这是本地准入集合的缺口，不将其推广成整个公开数据源没有这些动作；source8 DEV整源缺失
是split缺口。新source4不能填source8/10空格。下一批是否采集应依据全量覆盖重议，不能继续
沿旧6k的source4缺口自动扩采。

## 去重与来源认证

去重物理规则保持：PCA extent5%、完整28帧anchored joint21/query32最大坐标差2mm、
旋转1度、相同phase/validity、16part逐帧接触比例2%；同sequence/slot24tick内最后真实
保留代表或跨序列相同固定bin签名为候选，再作完整谓词判定，不用传递闭包，不按物体身份
放宽。跨bin/不同canonical CAD朝向存在漏检边界。旧61.23%是24帧内起点时间重叠率，
不能与内容删除率混称；全量最终同口径前/后数待完成。

历史全量native字段census给出0/11,197,332可认证，缺reset前独立TaskSpec、角色/来源、
reset/source时间或hash。记录`source_role=UNKNOWN`、`available_before_reset=null`；标签pose
SHA/mtime仅用于回溯。source4原生goal快照、H2O动作/verb标签存在，不自动成为原执行/reset/
来源绑定证书。完整逐窗denormalized账本未完成，现有sequence级证据及完整tuple索引可关联；
不把该关联说成完整逐窗认证产物已发布。

并行任务新96条/26,304窗口的显式任务级goal已独立只读复验PASS：source/hash/同clock时间/
reset前可得性、原生reset恢复、native21/778part、接触抽帧及全部event重算均核对。原分区见
[新96条报告](rigid-source-bound-pilot-20261023.md)，本轮未新采集。其source_role为
explicit_task_instruction、available_before_reset=true；goal证书不替event意图发证。
总catalog未生成；历史goal必须显式UNKNOWN，fresh须用既有reset_bound_sample，event均UNKNOWN；
禁止使用自身t+24 endpoint作部署goal。没有激活生产recipe或训练。

## 存储、复现与核验

| 项目 | 字节 |
|---|---|
| 新固定转换精确字节 | 182331193978 |
| 固定文件FS block取整 | 184756744192 |
| 完整峰值预检上界 | 228491466246 |
| 启动可用 | 271870013440 |
| 峰值后预测下界 | 43378547194 |
| 机主STOP时可用 | 64563957760 |
| 报告时可用 | 67027345408 |
| 硬线 | 32212254720 |

预算v1–v7、所有原转换attempt均保留，详见[精确预算](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/storage-budget-v7.json)。
部分旧display GiB字段陈旧，以bytes重算；实际fixed bytes已对账。环境graspenv/Python3.8.20、
NumPy1.24.4、Torch2.4.1/CUDA12.1、smplx0.1.28、trimesh4.10.1；[完整环境](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/environment-record.json)。
转换12workers、MANO仅GPU7；派生审计最多12workers（r4=8，优化版=4），不碰其他GPU作业。
seed登记：`{"dedup": 170, "full_manifest": 170, "preparation": 20261019}`，确定性编目不消费RNG；原Shadow表面采样20261019。
新96条probe seed198–203属独立分区，不重抽旧6k/512面板。

完整前去重manifest **SHA256 `4e40b4ec0bd58d390f8d5cee6047530d1832c3d5acb6080232251236d934c4c3`**：
[manifest](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/manifest.json) / [SHA文件](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/manifest.sha256)。
[TRAIN完整tuples](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/train-all.npy)、
[DEV完整tuples](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/dev-all.npy)、
[覆盖CSV](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/readiness-manifest-r1/phase-geometry-source.csv)均有manifest内文件SHA。
逐序列`receipts/`、64条批次`batches/`、实际转换/恢复命令`run-manifest*.json`均在同run目录。
原始ARCTIC295条与OakInk562条恢复的预算、输入、parity及NAS复制收据保留，未改只读源。

定向27项PASS；索引优化两次各14项serial/spawn/真实代表及持久化golden回归PASS，见
[检查清单](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/verification-summary.json)。r4/r5约696MB真实DEV公共前缀、去重代表账本、
native字段证据逐字节一致：[比对receipt](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/cache-output-prefix-check-r1.json)。
18个真实9source首尾reader样本PASS，但其legacy future-goal事实不当成部署认证。
全局verify有腾盘后历史cm-pointflow链接失效，未制造空文件过检查。

**暂停依据**：[STOP](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/STOP)、
[机主brief47b暂停收据](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/storage-stop-receipt-1792733578347140464.json)，13:32:58新增，
`automatic_restart=false`，信号策略SIGINT后暂停残留进程。未自行移除STOP或重启。
[本报告机器收据](../../../../../outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/hold-closeout.json)记录完成/缺失边界；0训练、0checkpoint、未删历史outputs。

后续并行写入的[正式brief47b空间保护记录](brief47b-space-hold-20261023.md)已确认：
新令优先于此前自动续跑授权，禁止新attempt、SIGCONT和自动重启。13:34:07容量快照
59.803GiB；一个新summary完整预算25.317GiB、metadata2GiB，加既有并行训练
剩余额度27.072GiB，扣完预计仅5.413GiB，距30GiB仍缺**24.587GiB**。
因此“当前还高于30GiB”不等于未来写盘已获准。当前不迁移、不删源数据，继续需要
机主新增容量或缩减范围的决定；具体大盘可用性与传输成本见该正式记录。

供机主审阅的后续命令（当前HOLD，未执行）：

```bash
# 仅在机主明确解除HOLD并确认容量/范围后，保留原STOP/receipt再恢复派生审计。
# 先检查可用空间与30GiB余量；使用新输出目录，既有partial不覆盖。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/summarize_full_rigid_data.py \
 --run outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1 \
 --workers 8 --output outputs/ptv3-shared-wam-v6/full-rigid-20261023-brief47-r1/full-manifest-r7
# 通过完整summary后再运行build_full_prepared_catalog.py及两份report renderer。
# 所有转换已完成；不需要重新转换，也不训练。
```
