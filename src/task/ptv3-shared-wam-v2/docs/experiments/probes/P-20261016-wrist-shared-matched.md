---
schema: ref2dex.probe.v2
probe_id: P-20261016-wrist-shared-matched
experiment_id: P-20261016-wrist-shared-matched
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: 18dfb0f9
claim_id: C1
hypothesis_family: HF-wrist-shared-matched
probe_index_in_family: 1
seed_pool: probe
seeds: [148, 196, 197, 199, 200, 201, 202, 203, 204]
decision_changed_if_positive: retain a direct811D control path and test aligned shared conditions in native grasp execution
decision_changed_if_negative: prioritize control training and feature readout before joint WAM adaptation
status: UNPROMISING
run_id: wrist-shared-matched-20261016-r1
---

# 同一腕部MLP有无shared的匹配对照

Result: UNPROMISING（限定本次冻结WAM、三组均值pooling及接入方式）：三个配对repeat中，aligned均降低TRAIN误差但增大DEV误差；冻结原MLP后线性shared残差也未改善主指标。
Decision: 保留811D直通MLP，不扩大本次pooling接法到native或joint长训；原shared actor的128D融合、控制头和监督目标须另做匹配定位，不将本结果推广为WAM无效。

Class: Decision。服务解释Inspire控制映射误差，以及预训练交互表示是否能改善机器人执行。
此前MLP与shared actor同时改变融合、目标、网络与优化，不能用其差距归因shared。
用户已授权执行同一MLP±shared，并利用本地/远端并行或分布式加速；本地仍保留两张空闲卡。

## 固定协议与能改变的决策

raw输入始终原811D（state87+已有goal24位置差3+当前721），同一strict
TRAIN93680/DEV10376窗口，同TRAIN-only归一化、原评估身份、半数startup/半数later且
phase内序列均匀抽样、model196/draw197、global512、fresh初始化、6000步固定终点。
仍只监督当前native腕部6D命令，L1按TRAIN标准化，不增加未来手参考或teacher动作干预。
原MLP隐藏1024/1024/1024/512保持；shared额外线性投影只加至第一层preactivation，
初始化全0，因此三臂初始模型和输出逐项相同。none臂使用零shared，但保留相同容量。

冻结共享7k WAM，在eval模式FP32/TF32对每个实际窗口提取shared，以有效mask分别对
scene/hand/goal求均值再拼接2304D；不把未来标签送入encode。每split独立缓存，
shared统计只在TRAIN拟合；shuffle在TRAIN/DEV内分别固定置换（199/200），不跨split。
pooling是本Probe具体readout限制，负结果不能否定其他token readout或整个WAM。

主三臂：none / aligned / shuffled，LR5e-4、WD1e-4。相同逐draw及模型初始化认证后执行。
附加两组none控制分别仅改LR为5e-5、或仅改WD为0.01，区分旧优化设置的影响；
这些不作为shared效用的配对臂，不混入主三臂因果比较。

主要读出为同一DEV256首控XYZ mean/p95与startup分项；TRAIN256检验拟合，另保存全部
TRAIN/DEV指标，不用更好看的all-window均值替换主面板。aligned相对none的mean和p95
均改善至少10%、startup不恶化，且优于shuffled，才记离线PROMISING并进入固定native；
否则UNCLEAR或UNPROMISING，保持探索结论边界。单seed不构成正式效用结论。

## 执行、资源与停止条件

本地GPU1/2/3/4连续分片并行提取同一冻结模型，GPU0外部作业不动，5/6/7空闲。
每分片<=1800s，完整缓存与训练<=8GiB，磁盘保留20GiB；非有限、身份/边界/覆盖不符即停。
合并必须每窗口恰好覆盖一次、checkpoint和pooling一致；原660及权重只读。
五组MLP可在GPU1–5并行（0外部、6/7空闲），每组<=900s/0.5GiB，仅保留固定终点。
阶段始末记录GPU利用率、显存和ETA；模型训练默认GPU。

远端四卡当前空闲，但portable迁移仍RECEIVING，不据目录存在宣称同数据可用。
先在独立remote run目录运行同形状MLP的1/2卡DDP工程吞吐检查（相同global512，<=300步）；
只决定执行方式，无学习结论。小MLP若DDP更慢，优先每卡独立对照；数据准备不阻塞本地。
不停止/改写既有上传、远端预训练或其他进程。正式运行保存本次实际代码提交和输入路径。

Decision Note：优先隔离shared条件而非新增模型路线。缓存冻结特征将神经计算摊销到所有
匹配臂，成本低于逐臂重复WAM；短DDP benchmark避免更大通信开销。主对照有信号再做
物理闭环及后续joint更新；无信号先定位融合/优化，不追加原joint长训。用户授权内执行，
没有改变最终claim或扩大数据权限，不需新增身份/分支。本轮不自动启动PPO。

入口：tools/run/probe_wrist_shared.py；模型/采样：src/ptv3_shared_wam_v2/wrist_shared_probe.py。
输出：outputs/ptv3-shared-wam-v2/wrist-shared-matched-20261016-r1/。

## 首轮结果后的最小重复（执行前）

固定6k首轮已结束：none TRAIN/DEV mean16.69/37.83mm，aligned14.89/42.61mm，
shuffled14.51/42.50mm；shared提高训练拟合但未获DEV收益。none完全复现既有MLP指标。
low-lr TRAIN/DEV35.90/40.19mm；high-wd17.30/36.95mm。以上均未选择DEV最好checkpoint。

Decision Note：单seed不足以判断负方向是否稳定，且原先两版混淆很大；复用同份冻结缓存，
对三主臂再做两个配对重复（model/draw201/202、203/204，shuffle199/200保持同一干预），
每臂仍6k/global512/原LR-WD，不追加单臂训练或挑种子。最多五卡同时运行，保留两卡空闲，
每组900s，预计整轮<2分钟；额外<=0.5GiB。只检查方向稳定性，不升级为Validation或否定
整个WAM。native正向准入门未过，本轮暂不扩大物理仿真。

独立只读review未发现使主三臂归因失效的实现缺陷，CPU重建6窗、future-label干预、
TRAIN-only统计、分片/合并、初始权重/首draw和实际实参均通过；这不替代科学效用检验。

## 原输入路径共适配的最小分离（执行前）

三个配对repeat的aligned均比none DEV mean差4.45–4.87mm，TRAIN均改善；shuffled并未
稳定更差。固定seed196 aligned权重推理时，移除shared反而使DEV42.61→58.84mm、
TRAIN14.89→64.26mm，说明原MLP路径已共适配，不能靠把新增输入置零恢复旧权重。
该推理遮罩是分布外干预，仅诊断依赖，不替代重训对照。

Decision：下一项最便宜的可判别问题是固定原MLP映射后，shared能否额外修正其误差。
冻结首轮none6k全部参数/预测，三臂none/aligned/shuffled只训练2304→6无bias线性残差，
相同TRAIN-only统计、draw197/global512/6k/LR5e-4/WD1e-4；none恒为原基线。残差初始0，
条件全0在训练结束后也必须精确恢复基线，base参数/梯度逐项检查不变。保持同DEV门，
正向才进native；负向只排除当前冻结pooling线性校正，不扩大否定范围。
GPU1/2/3并行，每臂<=900s，预计<1分钟，新增<=50MiB，磁盘底线20GiB。不启动joint/PPO。
入口tools/run/probe_frozen_wrist_residual.py；输出同run下residual-{none,aligned,shuffled}。

重复调度器记录写入曾发生并发rename冲突；五条已启动训练均正常COMPLETED，原失败
repeat-driver.json保留。仅补未启动的repeat203-shuffled，无权重或数值记录被覆盖。

## 最终结果（14个有界训练/控制运行均完成）

以下均固定6000步，未按DEV选点。主三臂使用三个配对model/draw（196/197、201/202、
203/204）；表内为三个run各自固定面板指标的均值，不把重复读取的同DEV256当成768条
独立样本。含控制的全部14行、每seed、startup、全量窗口指标见summary/results.json/csv。

| 主条件 | TRAIN256 XYZ mean mm | DEV256 XYZ mean mm | DEV startup mean mm |
| --- | ---: | ---: | ---: |
| none | 16.46 | 37.84 | 53.12 |
| aligned shared | 14.37 | 42.54 | 63.36 |
| shuffled shared | 15.13 | 40.52 | 61.74 |

aligned相对none的DEV mean三次分别增加4.78/4.45/4.87mm，约12.4%的平均退化；
TRAIN更准但DEV更差，且未优于shuffled。这个模式与额外条件过拟合/原路径共适配一致，
尚不能因果区分具体统计机制，不能从较低TRAIN误差宣布shared更有用。

单变量优化控制（seed196）：低LR5e-5的TRAIN/DEV35.90/40.19mm；高WD0.01为
17.30/36.95mm，基线16.69/37.83mm。降低LR显著拖慢固定预算拟合，高WD在此设置下
未造成明显退化；不能把原shared actor全部差距归给这两个超参数。

冻结原none6k MLP后的线性残差三臂：none DEV37.83/p95106.32，aligned38.16/110.72，
shuffled39.04/110.55mm。原MLP参数不变、无梯度及条件0精确恢复base的运行断言通过；
仍未达到主DEV增益门。此为单seed、无bias线性readout，不否定其他readout。

### 实现、资源和重放认证

原MLP四组raw归一化、第一批draw、新模型原net全部初始参数逐位一致；seed196 none
终点TRAIN/DEV16.69207/37.82655及loss完全复现历史MLP。三次配对各自全部初始参数和
第一批draw相同。独立review的身份、未来标签干预、统计和缓存检查通过。

独立smoke与四卡提取因TF32推理数值并非逐位一致：train/val特征最大绝对差
0.00518/0.00771，归一化后max0.00134/0.00203；检查的256窗经已训aligned模型映射后
腕部XYZ平均差0.0013/0.0024mm、最大0.066/0.040mm。该范围远小于本轮DEV退化量，
且主三臂始终共用同份冻结缓存；详见feature-replay-parity.json，不把任意latent绝对阈值
当作物理结论认证。后续native仍需独立实测执行审计。

四卡提取108.58s、每卡约3.9–4.1GiB；MLP各6k约19–21s，residual各约10–11s。
训练运行均有首步/阶段GPU利用率和显存采样，小head受host dispatch影响，未改用CPU。
本run约3.10GiB，低于8GiB；本地1–5现已释放，外部GPU0未操作。

远端约48GiB卡的同global512工程基准：单卡825.28 updates/s、双卡DDP365.79，
DDP仅单卡44.3%吞吐，因此选择每卡独立对照。该合成同形状benchmark只证明执行吞吐，
没有使用远端尚在RECEIVING的portable数据形成科学结果。远端benchmark进程已退出。

本轮code18dfb0f9，残差code d6e95c73；各运行process/manifest保存实际身份。未运行native，
未启动PPO，未更新/覆盖原始示范或checkpoint。结果只将本次pooling的两种接入方式记UNPROMISING。

### Limitations / future evidence

- 这不是正式Validation，不外推到所有对象、状态或整套WAM；固定DEV面板仅256窗，
  全量窗口指标单列，不以其替换主要门。
- 2304D由scene/hand/goal三组均值构成，未检验逐token注意力、其他WAM checkpoint或
  与shared等信息的原始H4/CAD/完整目标输入；正负归因均受该边界约束。
- 当前标签仍是teacher状态上的监督，未增加学生偏离状态的恢复数据，未证明闭环抓取。
- 原shared actor与腕部MLP之间的128D融合/decoder/八拍全动作监督差异尚未隔离；
  本轮只排除了“保持MLP其他设置后，当前冻结pooling条件自然带来收益”的近期预期。

残差独立review亦通过。残差none是本组三臂的实际对照：其head最终全0、终点与本轮
初始化面板逐位一致。与历史日志相比，全窗口1024批缓存base引起很小TF32批形状舍入
（主DEV mean一致，p95差0.000126mm；对象分项最大差0.11943mm），不声称历史各指标
逐位复刻。固定6k线性残差的负结果也不证明线性最优解无收益或shared没有有用信息。

汇总重导出（CPU文件/统计/绘图，无新增模型计算）：
`tools/audit/summarize_wrist_shared.py --source outputs/ptv3-shared-wam-v2/wrist-shared-matched-20261016-r1 --output outputs/ptv3-shared-wam-v2/<fresh-summary>`。
