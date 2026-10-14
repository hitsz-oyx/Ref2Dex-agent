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

Result: UNCLEAR，实施与检查中；未重启大规模联合微调。
Decision: 按用户ref2_1停止旧raw-native微调，先完成物理目标合同与matched R-only Probe。

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
