---
schema: ref2dex.probe.v2
probe_id: P-20261015-shared-current-continuation
experiment_id: P-20261015-shared-current-continuation
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: e3333e76
claim_id: C1
hypothesis_family: HF-shared-current-continuation
probe_index_in_family: 1
seed_pool: probe
seeds: [198, 192, 148]
decision_changed_if_positive: retain current information in original shared actor and test broader behavior
decision_changed_if_negative: do not assume the wrist-only MLP signal transfers to shared chunk training
status: UNCLEAR
run_id: shared-current-continuation-20261015-r2
---

# 当前721D在原shared联合训练中的匹配延续

已完成：两组追加1000步及三条val native均COMPLETED。masked/current的固定DEV256
action为0.40765/0.39792，XYZ mean61.73/60.87mm、p95171.33/162.61mm；两组飞机/球/杯
学生及自身工程重复均held0，各组teacher-positive，执行audit通过。预算内暂无行为收益，
保留UNCLEAR。实际action始终8拍，forecast为24拍，不受后续误写“默认24拍”影响。
后续7k对照及恢复语义更正见P-20261016-shared-action-horizon，本卡不再处于训练中。

Class: Decision。用户授权继续对照。此前完整721输入曾在GTtau旧R中带来约11%离线改善，
没有闭环验证；最近单拍腕部MLP+shared手指在TRAIN/VAL飞机held49/41，对照0/0，但
改了网络/目标/准入。目标是判别当前输入的行为价值能否迁回原shared训练，避免误认
腕部专训结果为原路线成功，也不重复把补齐输入描述成首次尝试。

最便宜且保持原目标的设计：两组都从同一shared7k权重初始化，WAM/R所有旧权重、
旧state/tau/action统计均相同；两组AdamW optimizer均重置，不是精确恢复旧optimizer。
相同追加1000步/global256单卡，原geometry5e-6/shared1e-5/R5e-5、WD.01、clip1、BF16、
TF32和forecast+8步15维geometry-lift q_PD L1；anchor0，无GTtau warmup，不改first6编码。
WAM/geometry继续接受action梯度和forecast监督；没有冻结/缓存WAM或新增腕部专用loss。

两组新增同一个721→128→128 GELU支路，加在原R state token上，末层全零使初始函数
等于shared7k。A=masked，TRAIN标准化后输入全置0；B=delta0，实际当前721D。A同样
训练支路bias等参数，容量相同。完整721数据来自同次补采660：contact/body/IG/当前
参考误差，禁止t+1/t+16参考分支；原WAM object goal24维持不变。
current normalizer只由strict TRAIN93680窗拟合，两个臂逐项相同；DEV10376与原shared
7k的DEV256身份/窗口准入完全一致。seed198初始化/逐step采样/PTv3随机，DEV192、native148。

执行前门：配对4步batch256 smoke，逐tensor initial actor/stat一致、初始audit loss一致、
两臂准入/原DEV一致。两臂native4步smoke均须通过own current721顺序/clock、未来不变性、
原有chunk解码和实际PD独立audit，再运行正式1000步。保存500/1000完整权重，不按DEV
挑物理测试点；旧checkpoint保留。最终分别在飞机149/球141/杯172 controls、K1、同
seed148上评估teacher/自身学生/自身工程重复。当前input由每行自己的delta0实时计算，
server接收值独立回显/对齐；没有teacher动作或teacher权重初始化学生。

判据：B相对A增加teacher-positive held45的序列数且两学生重复方向一致，局部PROMISING；
仅离线改善（mean/p95/startup分开）不作为抓取成功。若无行为收益，本预算UNCLEAR或
UNPROMISING，不能全局否定current信息；128D支路瓶颈、原8步目标及优化仍未分开。
旧腕部MLP是另一路正信号，不纳入这组匹配因果比较；多对象/多seed验证延后。

Decision Note：本轮优先验证同架构/目标下的信息干预。GPU6/7并行，现Shadow4/5与其他
已存在GPU0作业不动，保留1/2/3三卡空闲。两臂各<=4200s，smoke<=600s，native每组<=600s；
监督器绑定owned进程、非有限/磁盘<20GiB/新外来GPU消费者即停本臂，不杀他人进程。
总产物<=20GB，每正式训练臂<=7GB，不删除旧证据。所有代码先提交，detached工作树冻结，
精确git_commit由运行manifest/process记录。预计正式训练约30–40min，结束后自动native。

入口tools/run/probe_shared_current.py；产物outputs/ptv3-shared-wam-v2/shared-current-continuation-20261015-r2/。

启动记录：03c4fc58的r1在GPU计算前因隔离工作树缺PointWorld只读vendor依赖而退出。
保留r1与tmp/smoke-{masked,delta0}原始日志，不计方法负结果。r2连接同旧shared固定工作树
使用的PointWorld vendor，并使训练output basename包含run suffix，避免复用旧scratch日志。
实验输入、预算、权重、种子与两臂差异不变；修复后执行提交以r2/process.json为准。

## r2启动核验

冻结e3333e76，supervisor2589005，代码tmp/shared-current-code-20261015-r2。
配对4步batch256 smoke均完成，initial actor/stat逐tensor一致，原strict准入和DEV256
逐项相同，初始audit loss均0.2522353232，matched-smoke.json PASS。
geometry/shared/R组梯度均非零，保持原joint更新。两组native4步独立audit均通过，
max分别3.28e-7/2.23e-7，包含server own-current721回显/原chunk解码/实际PD检查。
只读独立review未发现关键语义或匹配问题，相关新增及原合同测试通过。
监督器已转入train1000阶段；正式离线和抓取结果尚未产生，不用smoke形成科学结论。
