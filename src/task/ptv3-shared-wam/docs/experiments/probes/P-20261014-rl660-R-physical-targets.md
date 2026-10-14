---
schema: ref2dex.probe.v2
probe_id: P-20261014-rl660-R-physical-targets
experiment_id: P-20261014-rl660-R-physical-targets
date: 2026-10-14
task: ptv3-shared-wam
branch: main
git_commit: 00a7e7ee8db74eb24d9c53688c1a09380a1ce9b3
claim_id: C1
hypothesis_family: HF-rl660-R-physical-targets
probe_index_in_family: 1
seed_pool: probe
seeds: [144, 145, 146, 147]
decision_changed_if_positive: use query-relative physical targets for next robot fine-tuning after native checks
decision_changed_if_negative: retain the contract correction but diagnose optimization and coverage without claiming label representation solves R
status: UNCLEAR
run_id: rl660-R-physical-targets-20261014-r1
---

# RL660动作标签表示对照：native / 相对PD / 相对几何目标

Result: UNCLEAR；三臂2k步dev XYZ mean/p95为native56.31/180.70、PD51.96/184.84、geometry51.25/186.15mm，均未过预定改善门。K4物理接线与独立审计通过；未重启大规模联合微调。
Decision: 保留统一当前时刻的PD目标合同和新微调模板；几何组因Euler分支不唯一禁止部署。R精度未解决，旧微调保持停止，不自动启动新联合长训。

## Decision Note 与研究动机

服务于MISSION自训练可执行底座，检验统一当前时刻相对目标能否降低GT-τ到实际PD的
拟合/泛化误差。用户已明确要求按ref2_1修订。旧微调在约14040步被用户停止，监督器
原生状态FAILED(signal15)是主动停止，不是方法失败；latest=step14000、best=step4500，
各约1.22GB，均保留；停止回执在原run/user_stop.json。旧部署守护4007948与执行队列
4092044取消，不在停训后自动启动旧K1方案。远端v2保持运行。

Class: Decision。最便宜判别是固定state87/GT-τ24、660语料与589/71整序列split，
只更新相同小R并比较三种标签，不重新训练forecaster或扩大R。不将历史单物体PDRetargeter
的12.5mm结果当作本50物体数据的预期保证。数据仍不是全部成功稳定抓取示范。

[PointWAM B.5](https://arxiv.org/html/2610.02840v1#A2.SS5)确认目标相对当前状态编码，
位置/手指作差、姿态为旋转向量，每horizon/坐标机器人统计标准化，推理时逆变换。
论文不证明本任务的全部误差来自表示。第一步XYZ新旧目标给定q后为仿射可逆；更换
标签不增加观测，不能消除原输入中的真实条件歧义。当前文档中的根因判断作为假设。

## 三臂固定协议

- native-command.v1：原记录applied[t:t+8]的12独立维；未来元素随其实际消费时q解释。
- pd-relative-current.v1：记录实际pd_targets[t:t+8]的12独立广义坐标，全部减同一个q[t]，
  包括手指；目标不是未来测量q，也不是q[t+k]作为模型输入。
- wrist-geometry-relative-current.v1：XYZ/手指同PD组，腕部旋转改为
  log(R(q[t])^T R(PD[t+k]))。URDF joint1..3为XYZ平移，joint4..6为内禀XYZ旋转，
  原点均为零；采用link6腕坐标，固定掌部旋转在其后。逐项合同测试对照实际资产。

三组模型ChunkRetargeter128D、2 decoder blocks、4heads、8queries完全一致，所有
可学习参数从相同seed144随机初始化，head零初始化；不把旧native权重当新物理R的
同函数暖启动。state/τ归一化复制保留的父微调TRAIN buffers，三组一致。输出分别用
110777准入TRAIN窗的每horizon/每坐标mean/std，std floor1e-6，同一统计算法，不用dev。
“相同normalization”指同方法/同TRAIN成员，物理单位不同不能复用native数值。

同sequence-uniform/half-startup sampler、同seed145批流、batch128，AdamW3e-4、
wd.01、clip1，每臂2000更新；all8 standardized L1等权，不叠加first/PD加权。只输入
state87/GT-τ；几何label和指标计算在CPU作小量解析转换，网络训练与推理用GPU5。
预打包仅在RAM，不额外写语料cache。TRAIN报告512由seed146选择；dev沿用原run固定
512索引（seed126）；它已被探索复用，不是独立holdout。

每500步报告TRAIN/dev第一步与all8恢复到实际PD后的XYZ mean/p95、第一步rotation/
finger p95、第一步及chunk clipping、startup/later。native未来步的离线恢复使用记录
未来q只作为控制器解码评价，不输入R；它是条件诊断，不代表闭环将访问相同未来状态。
物理组查询后冻结绝对目标，消费时用live q反算command，禁止把同一delta重复加到live q。
选点按dev第一步XYZ p95、再标准化all8 L1，step0也可被选；同时保留完整曲线与最终值。

若物理组相对matched native的dev mean/p95均至少改善20%，first/chunk clipping均<1%，
记局部PROMISING并进入GT执行门；否则为UNCLEAR，不把2k步或单seed结果当作方法无效。
若只改善TRAIN，下一步优先查泛化；若三组都欠拟合，按固定预算停止，不自动扩大模型。
几何组会丢Euler绕圈/分支；在全部准入TRAIN窗、每8步做exact-label→native→实际PD
复算。若广义PD最大误差>2e-6或出现clipping，它仍可作离线表示诊断，但禁止加载用于
native执行，避免把相同姿态误当相同servo路径。

## 资源、运行和停止条件

run_id=rl660-R-physical-targets-20261014-r1；1张空闲本地GPU5。保守计远端最多4张，
检查本用户/无法确认归属的本地活跃卡，合计<=6；不操作其他用户任务。R-only上限1800秒、
150MB；另预留工程联合2步与K4物理4步smoke各300秒，总新增GPU墙钟<=2400秒、
产物<=2GB（联合完整恢复点约1.22GB）。所有新run独立目录，不覆盖旧权重；聚合空间
预留受393GB主动停止线和400GB硬上限。非有限、foreign PID、标签roundtrip错误、
超时/空间越界停止自己进程，不自动重试或启动新的大规模联合训练。

工程smoke仅证明新训练/梯度/保存和执行接线；短4步不证明抓取。若Probe有正向信号，
仍需matched native执行与teacher校准，才决定新联合训练；本轮不自动长训。

## 实现入口

- src/ptv3_shared_wam/action_targets.py：版本化编码、query冻结、live转换。
- configs/robot_660_physical_targets.json：新联合模板，all8，PD目标，未启动长训。
- tools/run/probe_R_representation.py：三臂只改标签的GPU Probe。
- train_robot.py --init-forecaster-only：从保留robot权重只继承forecaster，重新初始化R和
  optimizer；不把旧native-R checkpoint静默解释成物理输出。
- native launcher/server/executor显式交换representation；旧schema默认native，旧first-only
  候选仍强制K1，新all8物理候选允许K4/K8。物理R-only候选有独立schema/controller/source/
  TRAIN统计核对，几何分支需额外roundtrip门。

## Limitations / future evidence

本轮不是正式Validation，不形成根因已坐实、R已解决或Cm有效结论。成功率、多seed、
独立新序列/物体、前馈forecaster误差与physics GT保持门留待正向Probe后。

## 工程修复记录：首次K4 smoke的日志注册

新物理执行首次到达真实Isaac第一控制，命令/PD一致性检查通过，随后因新审计字段
R_output_chunk未加入预注册日志字典而KeyError停止，已保留r1的partial/manifest/peer日志。
这不是控制策略负证据。修复注册R_output_chunk/R_query_q；仅在原2400秒/2GB总预算
剩余范围内以相同seed147、序列和4控制重跑受影响的工程smoke，使用新r2目录，不重跑
科学R-only对照或覆盖任何结果。不因工程重试扩大资源或自动启动联合长训。


## 实测结果与当前判断

科学Probe run `rl660-R-physical-targets-20261014-r1`，运行commit809a9e77（实现00a7e7ee），
COMPLETED，GPU5，125.43秒；实际约1.3–1.7GiB显存，小网络受采样/解析转换开销影响，
低于大模型GPU利用率属于本有界诊断的实际负载，不据此扩大网络或batch。
三组dev所选均为step2000（本预算最后一步）；无法以短预算判断最终收敛或泛化上限。

| label arm | TRAIN mean/p95 mm | dev mean/p95 mm | dev all8 mean/p95 mm | dev first/chunk clipping |
| --- | --- | --- | --- | --- |
| native | 59.17 / 236.82 | 56.31 / 180.70 | 72.45 / 226.05 | 0 / 0 |
| relative-PD | 54.84 / 223.02 | 51.96 / 184.84 | 83.43 / 232.09 | 0 / .513% |
| relative-geometry | 55.20 / 222.72 | 51.25 / 186.15 | 83.00 / 243.49 | 0 / .049% |

相对PD/geometry的first mean改善7.7%/9.0%，p95反而差2.3%/3.0%，都未达到mean/p95
同时改善20%的预定门，结论UNCLEAR；不能声称降到10–15mm或target定义已解释全部误差。
三组随机R初始dev XYZ mean均125.08mm，当前PD组仍在明显学习，不以2k步关闭此路线。
这些from-scratch、all8、seed144结果不能直接与先前warm-start/control-aware的43.02mm
作为纯label对照；本表matched native才是本轮控制。

全部110777准入TRAIN窗、8步exact-label回放：native最大PD差0，relative-PD为
1.91e-6（float32大角度相减/恢复误差），均无clipping；geometry最大广义坐标差
6.283185rad，即Euler绕圈分支不同，未获native资格。物理姿态等价不代表同servo路径，
几何组仅保留离线诊断权重；不能假装已经有无歧义的姿态到控制器转换。

独立工具 `tools/audit/audit_R_representation.py` 从原corpus重建state/τ/PD标签，以NumPy
控制器和独立Euler分支计算复算三个选点、TRAIN/dev预测及first指标，全部通过；最大XYZ
指标差<0.00005mm、clipping一致，报告 `independent-audit.json`。所有选点和曲线保留。

### 训练与执行工程检查

- `engineering-rl660-physical-targets-20261014-r1`，commit809a9e77，GPU7，真实PTv3共享
  512D×16与新PD-R从step14000仅继承forecaster，fresh R/optimizer；2步训练、验证和
  完整checkpoint保存完成，训练段约8.36秒。action→forecaster/R梯度非零，anchor→
  forecaster为0、→R非零；hand/scene均到PTv3/interaction/goal，scene不经hand head。
- 首次native工程r1因日志注册缺陷停在1控制，保留其失败证据；修复2bcafb76后只重跑
  受影响工程检查。`engineering-rl660-physical-native-20261014-r2`完成同seed147/同序列
  的4个控制、K4、teacher/GT/WAM三角色，33.53秒，显存观测峰值18,930MiB。
  `native/independent_audit.json` pass：PD、chunk clock、各自query q/H4、planned goal、
  被动command均核对；“查询冻结目标→实时q反解command”最大差1.19e-7。确实消费了
  chunk索引0..3，而非每tick重加delta。4控制teacher/GT/WAM均无held45证据，不能
  解读为抓取通过/失败，也不能用这个2步工程权重作策略收益结论。
- Task合同测试70项通过（00a7e7ee）；后续日志注册修复由真实K4重跑及独立数组审计验证。
  当前没有新的大规模联合微调进程。old latest/best再次用torch.load核对step14000/4500，
  完整可读；保留路径和大小在旧run的user_stop.json。

当前执行选择是relative-PD可逆合同；未来若继续研究R，需先区分现有预算下优化不足和
实际映射歧义，不能凭论文迁移的直觉直接扩大联合训练。更长R-only优化、初始化、容量、
额外状态/手点设计应逐一形成可判别变量；本轮不追加训练预算。
