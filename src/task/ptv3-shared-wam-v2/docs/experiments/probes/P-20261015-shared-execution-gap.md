---
schema: ref2dex.probe.v2
probe_id: P-20261015-shared-execution-gap
experiment_id: P-20261015-shared-execution-gap
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: d5f438e5
claim_id: C1
hypothesis_family: HF-shared-execution-gap
probe_index_in_family: 1
seed_pool: probe
seeds: [148, 149, 192]
decision_changed_if_positive: correct inference or reset mismatch before attributing failure to policy learning
decision_changed_if_negative: use stored TRAIN fit to distinguish optimization and information gaps from closed-loop drift
status: RUNNING
run_id: shared-execution-gap-20261015-r1
---

# Shared7k 训练状态拟合与执行输入一致性

Class: Decision。混控飞机teacher腕部+student手指held59、反向held0，TRAIN飞机仍held0。
服务当前目标：解释Inspire/Shadow抓取差异。先区分精确训练状态腕部本身拟合差、仿真
reset/推理合同不一致、以及拟合后闭环漂移；此分叉比继续动作表示长训更有决策价值。
最低成本为既有7k权重的离线GPU推理，无训练、无新仿真。

固定checkpoint为shared-native-grasp-20261015-r1/inputs/control-step7000.pt硬链接；模型
源码来自固定c1d2dca1工作树，脚本在执行前提交。使用原7k运行保存的workspace准入
快照重建相同顺序93680/10376窗口，确认其为源manifest准入窗口子集；不新增筛选规则。
TRAIN256按原sequence均匀/half-startup规则、seed192取样；DEV复用原validation.npy。
另完整评估TRAIN s1_airplane_pass_1和三条旧val飞机/球/杯的strict窗口，报告首拍PD XYZ
及startup<16/later分项。权重及normalizer不更新；batch16、FP32/TF32与原offline eval一致。

为排除执行合同问题，在同一TRAIN tick0样本上比较：单样本重复同seed148、单样本seed149、
2/16个重复样本、首样本加15个DEV样本。实际shared native server合推两学生行，是batch2；
原offline eval是batch16。eval模式关闭shuffle_orders，seed148/149仅检查重复性，
不将观察差异直接命名为序列化随机性。比较存储tick0的q/dq/hand/object/velocity、87D、
H4、goal24与原native reset；另保存重复两样本预测与原native预测差，有差先看输入再归因。
这些是实现诊断，不作为新seed验证或抓取成功证据。

执行前独立review发现VAL杯子strict准入仅144..171，tick0不在其中。拟合面板仍严格限定
28窗；reset一致性另从原始记录构造仅诊断用tick0，并标出不在strict membership，绝不将
该点混入拟合指标或训练。其他三个reset面板tick0均已准入。索引/采样/原native行映射
已独立审查，未见其他blocker。

Decision Note：此前窗口未见粗大label/clock错误，但未直接量化训练状态拟合与batch1/
batch16差别。GPU6已空闲，原训练仅GPU2/3和4/5，单卡评估后仍保留至少两卡空闲。
本轮<=900s/100MB、磁盘保留20GiB；进程/张量错误只停止本轮，不改已有checkpoint。
若推理/输入有明显漂移，先核实修复；若TRAIN也有明显腕部误差，优先学习问题；若TRAIN
准确而执行差，优先闭环覆盖/恢复。结果未出前不认定缺参考或数据少为唯一根因。

Limitations：TRAIN256不是全TRAIN穷举，四条轨迹是此前固定面板，无法代表全部对象。
随机序列化和批量差异均需量化，单个差值不自动证明导致抓取失败。新增当前参考输入的
对照仅在本诊断明确后设计，不自动扩展训练预算。

## 线性可预测性最小对照

27s离线结果显示TRAIN256 XYZ mean66.42mm、startup76.42mm，TRAIN飞机起步45.24mm；
batch1/2/16的所查首拍差<0.1mm，四条native起步状态/H4/goal均与源完全一致。因此先做
廉价闭式回归，检验控制目标信息和简单控制规律，而非再扩大joint训练。

固定相同strict全TRAIN93680/DEV10376、单拍native腕部6D label、同TRAIN-only标准化、
同ridge系数1e-3，不用DEV调参。A输入当前87D+已有物体goal24位置差3，B仅加当前
腕部q_ref[t]-q_real[t]6；没有参考t+1/t+16或未来实测。报告XYZ及Euler误差、tick0/1..15/later。
若A已明显优于shared，优先查模型/优化；若B比A明显改善，则当前参考条件值得进入神经
网络匹配对照；若两者均差，不把线性负结果当信息无用的证明。与WAM比较只作拟合诊断，
其loss/容量不同，不能据此声称WAM因果有害。

源几何finger有原生耦合差异，故仅用经reset首帧对齐核对的腕部6D参考，不将raw18D
全部称为native参考关节。每条读取时再核对首帧腕部6D。

CPU原因：这是96维以内矩阵充分统计及线性方程求解，没有神经网络forward/backward；
读取参考.pt仅为数据反序列化。<=300s/100MB，输出本run/reference-ridge，已有权重不改。
