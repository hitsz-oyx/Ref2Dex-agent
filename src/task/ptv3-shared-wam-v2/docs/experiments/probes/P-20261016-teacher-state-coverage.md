---
schema: ref2dex.probe.v2
probe_id: P-20261016-teacher-state-coverage
experiment_id: P-20261016-teacher-state-coverage
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 5
seed_pool: probe
seeds: [180, 181]
decision_changed_if_positive: prioritize relevant state coverage and sample-efficient feedback learning
decision_changed_if_negative: inspect effective coverage and control inductive bias before collecting more trajectories
status: PROMISING
run_id: teacher-state-coverage-20261016-r1
---

# 固定留出面板的教师状态覆盖对照

Result: COMPLETED，PROMISING（限定覆盖与采样能改善当前离线控制泛化，balanced通过预设门）。
Decision: 停止本轮采样搜索，保留expanded/balanced作为物理检查候选；先验证离线收益是否进入实际腕部/物体跟踪，再做匹配WAM收益比较。

Class: Decision。服务MISSION自训控制器与WAM可评价底座。用户已认可上一轮提出的覆盖
对照并要求持续推进。问题：旧杯类模型的泛化缺口能否由现有其他物体真实teacher状态减小？
最便宜方法是复用全660缓存，不仿真、不重新采集、不训练WAM。

## 协议与决策（训练前固定）

复用teacher-sequence-generalization的seen6849/held1707/unseen987身份。原杯类57序列
与扩大589序列对照，扩大训练106005行；原held及其[t-3,t+24]隔离带2538行全部排除，
DEV不进入训练。扩大只来自其他物体，不能把它等同新增杯类近邻状态或因果恢复覆盖。
初始模型使用旧step0随机权重，不使用teacher actor权重；原teacher RMS与原seen-only
目标mean/std全部固定，防止归一化改变损失权重。1442→1024→1024→1024→512→12，
relative wrist6/absolute active fingers6，L1，AdamW3e-4/WD1e-4/clip1，24k更新，
12k后cosine到3e-6。sequence-uniform、half-startup、batch512，初始化180/抽样181。
原始组重新运行并与旧终点比较；本轮相同seed仅用于匹配，不是独立重复。

固定同组sequence-macro腕部XYZ mean为主指标，两留出均改善>=10%，p95不恶化>5%，
旋转p95不恶化>10%，且原seen拟合<=5mm或显著恢复，则标PROMISING并进入小闭环/覆盖
后续判别。均无改善且拟合合格时，本次扩大配方UNPROMISING；若扩大组训练欠拟合或
原杯类曝光显著稀释，记UNCLEAR，不能由此否定覆盖假设。保留全训练拟合与分阶段指标。
中期曲线只描述，不使用DEV选点。检查全1442及物理反馈子块的近邻覆盖以解释数据是否相关；
近邻不保证控制等价。后续配方另记Decision Note，不做无界参数搜索。

## Decision Note

关键证据：原学生1000→24000步seen18.40→0.56mm，unseen41.51→42.44mm；
冻结WAM六臂未改善。原cache教师回放max0.000795mm，故完整输入足以再现教师标签。
root选择先比较已有数据覆盖，避免重复读出/容量搜索。目标是定位而不是用大量示范替代
少样本目标；新Shadow v3早期正信号属于独立Task，不覆盖本轮Inspire结论。

## 资源、运行与停止

本地GPU0，启动时空闲，GPU6/7保留空闲；不干预GPU1–5既有任务。每臂1200秒/512MB，
本轮两臂+独立smoke/数组审计合计<=45分钟/2GiB；磁盘至少20GiB。
工程smoke2步不进入科学结果。训练记录GPU显存/利用率/ETA。非有限、外来consumer、
成员泄漏、原始组不能复现或预算超限停止自己的run。TMPDIR=项目tmp。
实现入口tools/run/probe_teacher_coverage.py，产物outputs/ptv3-shared-wam-v2/
teacher-state-coverage-20261016-r1/{original,expanded}，运行manifest记录实际代码commit。
保持main，不新建分支，不覆盖其他会话修改或旧产物。

## Limitations / future evidence

单训练seed、反复使用DEV、特权完整教师参考、固定杯类面板；不证明跨手型或闭环收益。
全量抽样减少原杯类曝光，负结果须检查欠拟合；更多其他物体可能不覆盖关键反馈状态。
若信号明确再做物理执行和后续匹配WAM对照，多seed正式证据延后。

## 首两臂结果与等标签预算后续 Decision Note

original运行代码743f7067，expanded代码c3b3b4cd（仅增加audit工具，训练源相同），
24k耗时67.4/69.0秒，GPU0约2.15GiB，训练利用率波动1–56%，吞吐约350更新/秒正常。
原模型终点逐tensor与旧24k相同。独立新进程checkpoint重推、原标签重算、成员和统计
检查全部PASS，保存预测重放差0。held/unseen为32.652/42.436→22.634/27.368mm，
p95为76.961/101.488→46.730/75.654mm。扩大组原seen7.012mm、全训练10.067mm，
未达到原seen<=5mm资格门，因此整体暂记UNCLEAR，明确记录两留出均改善的正向信号。
不能追溯放宽资格门，也不能将其表述为已解决控制。

标准化root参考误差/旋转误差、实测线速度/角速度与指尖contact共40维的TRAIN最近邻
距离中位比值held0.793/unseen0.843；65.5%/58.2%的行近邻改善>10%。全1442维的
中位比值约1.000/0.996，说明相关子空间覆盖变化更明显，不证明控制因果等价。

新增问题：收益是否需要更多标签，还是相同标签数下更广序列覆盖就有价值？
选择追加diverse一臂：从合法expanded中按序列均衡、不读标签选择6849行（seed181），
每轮随机序列顺序、逐序列取一随机候选直到预算用完，排序恢复cache身份。仍24k、相同
初始化/归一化/模型/更新。原seen面板现在只是固定杯类对照面板，不能冒称全部见过；
另报真正训练集all_fit_states。样本来自589序列而非57，不能称相同示范轨迹预算。
预设若两留出较original改善>=10%、p95不恶化>5%、rotation不恶化>10%、实际训练集
macro<=5mm，支持状态多样性优先；否则据覆盖与拟合边界记UNCLEAR，不追更多步。
本轮最多新增1200秒/512MB，在原45分钟/2GiB总预算内；结束后再决定后续，未启动闭环。

## 等行数结果与最终混合采样 Decision Note

diverse代码634a3531，24k约69秒，真实训练macro0.146mm；两留出macro26.696/36.661mm，
相对original改善18.24%/13.61%，但unseen p95=111.495mm，比101.488高9.86%，未过5%
尾部门，记UNCLEAR。原seen杯类面板31.659mm不是训练拟合误差，该面板只有部分行训练过。
三臂都共享原始杯类标签统计，diverse是相同梯度训练行数，不是严格相同全部标签信息预算；
589序列也不是57条轨迹的等示范预算，不能直接宣称few-shot效率改善。

当前决定：expanded增加相关覆盖但杯类曝光稀释，diverse牺牲了部分尾部；最后一次
balanced对照保留106005全部合法行，只把每批512改为256原杯类+256其他物体，每部分
各128startup/128later，内部仍sequence-uniform；两个确定性抽样流均seed181。
原初始化/统计/模型/24k更新不变。测试是否同时保留原seen<=5mm和两留出mean改善>=10%、
p95不恶化>5%、rotation不恶化>10%。否则保留最便宜的已知观察，不追加同族超参搜索。
新增最多1200s/512MB，总本轮四臂+smoke/audit<=45分钟/2GiB；GPU0，6/7空闲。
它改变采样分布而不改变数据集合；结果用于选择后续覆盖/反馈方向，不证明WAM收益。

## 完成结果与归因

四臂均固定24k，无DEV选点。balanced代码0bd40c30，70.45秒；完整表单位mm，
固定杯类面板在diverse中仅661/6849行实际参与训练，不把它叫该臂训练误差。

| 训练设计 | 训练行/序列 | 固定杯类原seen | 隔离macro | unseen macro | 隔离p95 | unseen p95 | 实际全训练macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| original | 6849/57 | 0.564 | 32.652 | 42.436 | 76.961 | 101.488 | 0.564 |
| expanded | 106005/589 | 7.012 | 22.634 | 27.368 | 46.730 | 75.654 | 10.067 |
| diverse | 6849/589 | 31.659 | 26.696 | 36.661 | 60.695 | 111.495 | 0.146 |
| balanced | 106005/589 | 1.387 | 25.060 | 28.478 | 51.004 | 72.648 | 12.909 |

balanced两留出macro分别改善23.25%/32.89%，p95均改善，旋转p95由0.1504/0.1996
降至0.1110/0.1474rad，原seen1.387mm<5mm，满足本臂运行前规定的继续门，记PROMISING。
这不是所有物体的通用控制资格：balanced全训练macro仍12.909mm，非杯类学习不足；
不能把对目标杯类重采样的正信号扩成全589序列策略更优。expanded在两留出平均值更好，
balanced保留更好的原杯拟合且unseen尾部略低，不以单个指标宣称一者全面胜出。

这轮改变的决策：当前失败对真实训练状态覆盖/抽样明显敏感，优先改数据分布与反馈学习，
而非再次加大网络或重复冻结WAM读出搜索。但它仍没有单独区分局部近邻覆盖、其他物体的
共享规律与优化正则化效应；不把最近邻距离相关性写成已证实因果机制。
diverse均值改善但尾部退步，不能总结成“只要更多序列就好”，更不是少示范能力已建立。
本轮没有WAM干预，既不证明WAM有效，也不解释完先前WAM支路负结果。

## 最终核验、资源和下一步边界

`tools/audit/audit_teacher_coverage.py`在独立进程重读四个checkpoint及原cache行，
12组固定面板预测重放maxabs=0；标签macro/p95重算一致。原始组与历史24k权重逐tensor
相同，原seen身份保持，所有held支持最近距离>=28帧，训练/DEV序列不交叠；四组归一化
完全相同。diverse行数/去重/准入、balanced数据集合与首batch256杯/256其他、
256startup/256later全部PASS。9项划分与采样合同测试通过，仓库verify通过。

正式四臂单卡训练共275.61秒；每臂约67–70秒、2.10GiB显存，采样利用率最高45–56%，
小MLP由host调度限制，吞吐稳定，无非有限或OOM。工程smoke、GPU近邻与重推另计，
总产物约66.6MiB，明显低于2GiB；GPU0已释放，6/7全程空闲，其他任务未操作。
实际训练commit分别743f7067/c3b3b4cd/634a3531/0bd40c30，最终audit代码0bd40c30；
各manifest独立可追溯。

产物根`outputs/ptv3-shared-wam-v2/teacher-state-coverage-20261016-r1/`。
`audit-final/results.json`保存最终复算与覆盖统计；`summary/comparison.png`及PDF保存学习
曲线和尾部对照。`summary/results.json`的seen字段表示固定原杯面板，不代表diverse训练集。

下一项最小Decision是同一固定DEV序列的native物理执行：original与expanded/balanced，
相同参考、状态、频率、动作解码，量实际腕部/物体参考误差、接近、持物与掉落；完整1442D
教师参考必须显式标为辅助输入，不得称纯物体轨迹策略。正式切入前补齐此scratch控制头的
native执行适配与命令审计。本轮到此结束，不再增加离线步数/比例，不启动PPO或全量采集。
