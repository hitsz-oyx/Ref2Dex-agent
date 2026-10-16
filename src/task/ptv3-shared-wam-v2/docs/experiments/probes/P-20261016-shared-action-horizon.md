---
schema: ref2dex.probe.v2
probe_id: P-20261016-shared-action-horizon
experiment_id: P-20261016-shared-action-horizon
date: 2026-10-16
task: ptv3-shared-wam-v2
branch: main
git_commit: af861865
claim_id: C1
hypothesis_family: HF-shared-action-horizon
probe_index_in_family: 1
seed_pool: probe
seeds: [198, 192, 148]
decision_changed_if_positive: use current721 plus first-action supervision as the next matched shared route
decision_changed_if_negative: retain full-chunk supervision for current-input shared training and investigate input bottleneck separately
status: UNCLEAR
run_id: shared-action-horizon-20261016-r1
---

# 当前输入与动作监督跨度：执行结果及语义审查更正

Class: Decision。服务C1。预定四臂masked/delta0当前输入×1/8拍动作监督，均从shared7k
权重起步、原WAM/R联合训练及forecast保持。动作输出始终8×15D，forecast另有24拍。
8拍臂原先从shared-current的1k点恢复到7k，1拍臂从同一7k parent fresh optimizer训练7k。
数据strict TRAIN93680/DEV10376、原DEV256、seed198/192/native148；global256，原三LR。
GPU4/5单拍、6/7八拍，GPU1腕部面板，2/3空闲。每臂14000s/7GB、磁盘底线20GiB。
原运行记录和checkpoint保留，不将预算停止当作算法失败。

## 资源恢复与实际代码路径

02:10前后原continuation两臂在step4000之后触及7GB artifact budget停止，随后从完整
step4000恢复模型/Adam/RNG，改为稀疏validation/checkpoint cadence，在
continuation-{masked8,delta08}-sparse完成7000。模型/数据/学习率未因资源恢复改变。

single-masked1在step5000、single-delta01在step6000后也进入sparse恢复，但其实际执行
源码发生了下述监督语义变化。原始日志、失败和恢复的process.command均保留。

## 审查更正：原8拍判无效有误，单拍尾段有真实错配

此前04:55审查把forecast24拍混同为action24拍，错误判定未写action_supervision_steps
的continuation为24拍，并为此重跑fresh8。实际process.command所指旧固定树中，actor
只有8个action_queries、action_scale为8×15、标签也是8×15，robot_loss直接对全部8拍
求均值。新版缺省取action_error.shape[1]，也等于8。因此撤销原continuation的
“24拍INVALID_IMPLEMENTATION”判定，shared-current的1k不存在所谓24拍污染。

fresh8仍是有效8拍运行，但属于额外重复，不是修复24→8所必需。原结果不删除，也不
挑其中更好的一组冒充唯一预设执行。两个fresh8最终manifest均COMPLETED。

真正的实现错误发生在single稀疏恢复：两条single-*-sparse/process.json都指向
`tmp/shared-current-code-20261016-cont7k-sparse/src/task/ptv3-shared-wam-v2/tools/run/train_robot.py`。
该固定树的robot_loss完全不读取action_supervision_steps，仍监督全部8拍。因此：

- single-masked1-sparse：前5000步单拍，后2000步八拍。
- single-delta01-sparse：前6000步单拍，后1000步八拍。

这两个7k端点对“完整7k单拍”标INVALID_IMPLEMENTATION；作为混合协议结果保留，不能
据其归因单拍监督优于8拍。manifest写1和provenance里的主仓库HEAD均不足以认证实际
执行语义。旧git_commit按REF2DEX_RUN_ROOT主仓库取值，不能替代实际导入源码核对。

CPU最小复现抽取实际robot_loss，使用(1,8,15)首拍误差0、后7拍误差1、配置监督1拍：
旧sparse函数loss0.875，新函数0；新版缺省和显式8均0.875，显式24越界报错。native日志
R_output_chunk实际为(controls,2,8,15)，24×33是R_tau。独立只读review一致。
审计入口`tools/audit/audit_horizon_runtime.py`，更正证据为
`outputs/ptv3-shared-wam-v2/horizon-runtime-audit-20261016-r1/correction.json`。
原results_summary.json保留不覆写，其24拍和invalid分类已由本更正取代。

## 最终离线读出：区分有效8拍与混合协议

| 运行 | 实际action监督 | XYZ mean / p95 mm |
| --- | --- | ---: |
| continuation-masked8-sparse | 8拍7000步 | 59.76 / 173.90 |
| continuation-delta08-sparse | 8拍7000步 | 55.91 / 163.36 |
| fresh8-masked（额外重复） | 8拍7000步 | 60.09 / 174.50 |
| fresh8-delta0（额外重复） | 8拍7000步 | 56.13 / 164.95 |
| single-masked1-sparse | 1拍5000 + 8拍2000 | 59.07 / 165.41 |
| single-delta01-sparse | 1拍6000 + 8拍1000 | 55.14 / 161.31 |

这些端点不能再作为完整四组“1拍vs8拍”对照。有效8拍配对和额外重复均显示delta0
在离线首控XYZ上有收益；是否改变监督跨度，当前证据不足。

## 已执行native：只对应各自实际训练协议

seed148、飞机149/球141/杯172 controls、K1。指标partial_hold45_only_no_placement。
完成面板的independent wiring audit通过，只证明所查执行合同，不能认证训练目标。

| 权重 / 对象 | 学生max held | clipping | 状态 |
| --- | ---: | ---: | --- |
| fresh8-masked / 飞机、球、杯 | 0 / 0 / 0 | 41.6% / 8.5% / 19.8% | 完成 |
| fresh8-delta0 / 飞机 | 0 | 13.4% | 完成 |
| fresh8-delta0 / 杯 | 60 | 0% | 完成，有保持 |
| fresh8-delta0 / 球 | — | — | query87越出固定PTv3 workspace |
| single-masked混合 / 飞机、球、杯 | 0 / 0 / 0 | 12.8% / 0% / 4.1% | 完成 |
| single-delta0混合 / 飞机 | — | — | query88越出固定PTv3 workspace |
| single-delta0混合 / 球 | 0 | 22.0% | 完成 |
| single-delta0混合 / 杯 | 117 | 0% | 完成，不能归因单拍 |

当前仍UNCLEAR：有当前信息的8拍策略在杯子出现局部保持信号，但其他对象仍失败/越界；
撤销“完整单拍略优于8拍”的归因，不进入基于该结论的多seed Validation。

## 最便宜的有效后续

四臂均有尚未错误恢复的共同4k checkpoint。已存离线mean/p95：原8拍masked60.17/174.28、
delta0 57.85/170.52；单拍masked56.28/170.82、delta0 53.61/164.65mm。优先补核这四个
前缀的初始化/样本/预算匹配并做固定闭环，成本低于重跑全部训练；本次仅审查，不新开长训。

完整输出仍在`outputs/ptv3-shared-wam-v2/shared-action-horizon-20261016-r1/`。
后续residual集成仍需显式adapter及fresh zero-residual检查；现有离线apply_residual(base,0)
恒等检查不等于完整server集成，本Probe未启动PPO。

## 共同4k固定闭环（执行前）

本轮用户询问下一步，root执行最低成本的共同4k qualification，不增加训练。
masked8/delta08取原continuation的step4000，masked1/delta01取原single的step4000，
均早于错误的单拍sparse恢复。入口screen_horizon_prefix.py先检查四组初始actor和
TRAIN current统计逐tensor相同、同父shared7k、数据准入/DEV身份相同、结构/三LR/
优化器设置/seed198/global256一致；8拍0→1k沿用原paired运行的初始化记录。
这验证已记录的关键链路，不将工程matching等同正式随机化Validation。

四臂分别GPU1/4/5/6，同时各跑飞机149/球141/杯172 controls、seed148、K1、
teacher/自身学生/自身工程重复，使用已通过的e3333e76 current721 native bridge。
bridge只推理，不调用训练loss，故能加载真实1拍和8拍相同结构的checkpoint。
每组实际PD/own状态/721时钟独立审计，workspace退出单独记录，不补0或自动重试。
同一输入下1拍增加teacher-positive held45序列且重复方向一致才考虑短跨度；无行为
改善则不因离线均值更低改路线。两种输入都失败时保留数据覆盖/执行状态分布疑点，
不直接转PPO。当前只做三对象小Probe，没有多seed成功率结论。

每臂三组、每组<=600s物理deadline，整轮<=2400s/4GB，磁盘保留20GiB；已有GPU0
作业不动，GPU2/3/7空闲，预算合规。不覆盖7k/4k权重或原错误汇总。
产物outputs/ptv3-shared-wam-v2/horizon-prefix4k-screen-20261016-r1/，实际运行代码由
process.json记录。若matching失败则在启动仿真前停止；执行失败只影响本组并保留日志。

## 2026-10-16 会话接续：当前721D首控拟合定位（执行前）

Class: Decision。服务主线当前721D→shared R，而非继续扩展监督跨度。
问题：有效fresh8的721D收益为何没有推广为三个对象的稳定抓取？
现有7k轨迹中，同源初态完全相同，current飞机/球首拍PD XYZ误差仍46.49/56.95mm；
第16帧腕部相对同次teacher偏122.3/134.0mm，早于teacher首次表面距离<=1cm的42/27帧。
因此先用冻结模型推理，区分stored TRAIN拟合不足与跨序列泛化/闭环恢复问题。

最低成本：fresh8-masked/delta0固定step7000，同源strict93680/10376准入和既有
shared-execution-gap面板（TRAIN balanced256/DEV fixed256及原四条序列），分别报告
startup<16/later首控XYZ；不更新参数、不新增示范、不改变DEV、不用纯1拍结果选主线。
模型使用已通过native执行的固定e3333e76代码，逐项核对snapshot membership和DEV身份。
若TRAIN也差，下一步优先对齐actor拟合目标/优化；若TRAIN准而DEV差，优先覆盖/泛化；
若两者均准而执行差，再优先学生偏离状态的恢复数据。此小面板不能排除其他因素。
GPU1当前空闲，GPU0外部任务不动；<=900s/100MiB，磁盘保留20GiB，无新训练/仿真。
输出 `outputs/ptv3-shared-wam-v2/current-policy-fit-20261016-r1/`。

### 本次实际结果与下一步决策

GPU1冻结推理完成，用时54.88s、显存采样约1.14–1.17GiB，GPU11–35%；
batch16和逐窗CPU读数限制短诊断吞吐，无训练/仿真，也未占GPU0。严格准入、DEV身份、
模型strict load均通过。复算DEV均值与原run相差<0.01mm，微小重放数值差不影响判断。

| 模型 | 同一TRAIN balanced256 mean / startup mm | 同一DEV256 mean / startup mm |
| --- | ---: | ---: |
| fresh8-masked 7k | 58.08 / 65.50 | 60.10 / 103.64 |
| fresh8-delta0 7k | 56.05 / 63.31 | 56.12 / 95.19 |
| 既有current-wrist MLP 6k（诊断参照） | 16.69 / 11.15 | 37.83 / 52.41 |

TRAIN面板startup128/256，DEV仅21/256，不能用TRAIN/DEV总体均值相近否定泛化问题。
MLP与shared使用相同评估身份及物理首控XYZ单位，但结构、融合、loss、采样和优化均有
差异，因此这里只证明这些存储状态在已有当前信息下存在更精确的可学习映射，不能认定
WAM、128D支路或8拍监督中某一项是唯一原因。shared新721D权重仍有明显TRAIN拟合残差，
应先排查控制映射的学习方式，而非单纯把现象归为学生闭环偏移或新增数据不足。
本轮保持UNCLEAR，不升级为因果或跨对象正式结论。

保存轨迹独立重算五条完整fresh8运行的held均与原result一致，delta0球按partial单列。
current飞机/球首拍误差46.49/56.95mm；第16帧与同次teacher腕部偏差122.3/134.0mm，
teacher首次表面<=1cm分别第42/27帧。tick0与source所有三行q完全相同，因此首拍误差
不由已发生的学生分布漂移解释；后续相对teacher位置偏差只是描述量，不当作动作标签误差。

4k前缀结果已完成且完成组audit通过：纯1拍current杯子shared/repeat held92/91，飞机/球0；
8拍current飞机0、球/杯workspace退出。纯1拍masked球shared90但repeat0且clipping高，
方向不稳定；不挑单条成功据此变更主线。原7k杯子60/61和4k单拍92/91不是matched步数比较。

Decision：保留当前721D作为后续主线输入，不追加相同joint长训或围绕单拍扩展矩阵。
下一项最小干预建议从同一shared+721D起点做joint更新与冻结WAM仅训R的短配对，保持
当前输入支路、8拍目标、同样本和R优化器设置；先看TRAIN/startup精度是否实质改善，
有改善再复测固定闭环。若冻结仍不改善，再单独检验输入融合/控制目标，不同时改多项。
MLP的DEV残差也表明泛化/覆盖仍未解决，后续需另核；本次没有启动上述训练或PPO。

证据：`current-policy-fit-20261016-r1/result.json`、`panels.json`、`fit-comparison.json`、
`native-trace-analysis.json`及`approach-drift.png`；完整路径均相对本Task outputs。

### 用户追问后的比较范围纠正：并非同一MLP仅增加shared输入

重新核对实际SharedQueryActor和wrist MLP源码：MLP直接读811D（state87+goal24位置差3+
current721），隐藏1024/1024/1024/512，输出当前native wrist6；shared actor分别将state87
和current721编码成128D并相加为单个state token，shared768D token也投影128D，再经
两层128D TransformerDecoder、8个action query输出8×15 geometry-lift编码动作。goal经
WAM进入shared，而非MLP的显式3D输入；它不是把原MLP完整保留再增加一个可忽略的输入。

其他同时变化：MLP监督标准化单拍native腕部命令；shared监督8拍geometry-lift全动作并
保留hand/scene forecast。MLP AdamW LR5e-4/WD1e-4、batch512、fresh6000步；shared
R LR5e-5/WD1e-2、batch256、从旧shared7k追加7000步，WAM同时更新。两者都按半数
startup/半数later且phase内序列均匀采样，但seed、batch与实际draw不同；不能笼统说
采样分布完全不同。数据池、评估身份、物理XYZ单位相同，只支持诊断比较。

收紧前述下一步建议：joint vs冻结WAM仅训R只检验联合更新影响，不能回答“在相同MLP
上增加shared是否有益”。优先设计同一腕部MLP的±shared配对：保持811D原始输入直通、
相同腕部6D标签/归一化/隐藏宽度/初始化/抽样/优化预算，仅增加可忽略的shared条件。
先冻结WAM隔离特征效用，有信号后再检验联合更新。新增支路零初始化可保证初始预测
一致，不保证有限训练后误差必然不增。本轮仅核对与纠正建议，没有启动新训练。
