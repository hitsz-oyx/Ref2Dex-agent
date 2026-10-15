---
schema: ref2dex.probe.v2
probe_id: P-20261015-current-wrist-mlp
experiment_id: P-20261015-current-wrist-mlp
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: f816b3d5
claim_id: C1
hypothesis_family: HF-current-wrist-mlp
probe_index_in_family: 1
seed_pool: probe
seeds: [196, 197, 148]
decision_changed_if_positive: add captured current teacher information or repair wrist optimization before further joint training
decision_changed_if_negative: prioritize generalization or missing temporal intent over another equivalent joint training extension
status: UNCLEAR
run_id: current-wrist-mlp-20261015-r1
---

# 当前腕部监督 MLP 配对 Probe

结果摘要：两臂6000步已完成，20.13/19.97s，总产物555MiB。相同起始面板预测完全一致、
首batch draw逐项相同；只读独立review未发现关键输入/标签/单位/时钟/匹配错误。

| 面板 XYZ mean mm | A state+goal | B 再加当前721D |
| --- | ---: | ---: |
| 与旧shared相同TRAIN256 | 20.56 | 16.69 |
| 与旧shared相同DEV256 | 42.60 | 37.83 |
| DEV256 startup21 | 76.56 | 52.41 |
| 全TRAIN93680 | 26.22 | 20.98 |
| 全DEV10376 | 48.23 | 40.13 |
| 全DEV startup971 | 85.09 | 56.93 |
| VAL飞机149 | 42.74 | 34.79 |
| VAL飞机 startup16 | 41.73 | 19.34 |

原shared7k在同TRAIN256/DEV256是66.42/62.70mm。A虽未给完整teacher当前信息，已大幅
降低误差；说明原设置仍有可改善空间，但这里同时改了网络/损失/学习率/goal显式入口，
不能单独宣布学习率或shared压缩导致失败，更不能把离线改善等同抓取成功。
B相对A：DEV256均值改善11.21%、p95改善17.13%；全DEV均值改善16.78%，起步改善33.09%。
未达预设DEV均值>=20%的强信号线，因此当前信息增益标UNCLEAR，而非坐实主要information
mismatch。起步增益值得保留，但两臂仍有明显TRAIN/DEV差距，B全DEV40mm也未跨过20mm。
新增721D含当前参考误差/接触/body/IG等，不将增益单独归因某字段。

下一步决策：停止同预算拟合扩展，不因原native/qPD两臂失败放弃监督腕部路线。
全DEV起步改善33%但未跨过整体20%强信号，仍保持UNCLEAR。root选择最小闭环筛查，
判别“降低起步腕部误差是否足以恢复接近/抓取”，再决定是否修改shared联合训练；
该筛查不改变预设offline强信号标准，也不将起步改善升级为整体正结果。

## 离线之后的最小闭环筛查（执行前固定）

复用原shared7k手指，仅替换当前腕部6D；teacher/A腕部+自身shared手指/B腕部+自身shared
手指三行，30Hz K1/seed148。每行独立测量state87、H4和同当前delta0，无teacher动作混入。
该部署是组合策略，不能称为原shared策略恢复。t=0未来reference置零负控制保证delta0
不受未来手参考影响，恢复后才step；官方teacher仍用其原本1442输入作为对照。
先4步工程smoke+独立审计；通过后GPU6的VAL s10_airplane_pass_1/149controls与GPU7的
TRAIN s1_airplane_pass_1/195controls并行，各最多600s/1GB。只做两条，不扩扫ckpt/seed。
held45且teacher-positive作为局部正信号；否则按最小surface gap和接近时序决定是否值得
后续修原joint学习。独立审计重构混合/PD、own-feature、CPU重放MLP（设备浮点容差1e-4
command，原command和PD审计仍1e-6），不宣称完整721每字段的独立语义认证。
加载MLP在fork_rng内建网络，不改变teacher/environment RNG流。保存完整前后命令与输入。
如审计/未来不变性失败，只修关键实现再跑受影响部分，不将失败计作方法负证据。

运行PID2492215/2492216均退出，GPU6/7已释放。训练快于首次资源轮询，未留存训练中
nvidia-smi利用率采样；实际运行设备/代码由各manifest和launch.json记录。后续同类快速
Probe应在step1内保存显存/利用率，不能把完成后的0%当训练效率。原Shadow4/5保持运行。

Class: Decision。服务解释Inspire/Shadow差异。混控定位腕部，stored TRAIN256误差66.42mm，
所查reset和batch差不足0.1mm；线性加入当前腕姿误差DEV82.13→82.53mm没有改善。
最低成本使用补采strict相同93680/10376窗，直接拟合当前native腕部6D，不再跑WAM。

A=当前state87+已有物体goal[t+24]位置差3；B=A+同一执行帧捕获的DExplore delta0 721D。
没有增加未来手参考t+1/t+16，无tau。检查源current-only/future-invariance审计及frame_tick。
A的额外721维置零，两臂完全相同811→1024→1024→1024→512→6 ReLU网络和初始权重、
零输出头、TRAIN-only输入/标签统计、AdamW LR5e-4/WD1e-4、归一化L1、batch512、
seed196初始化/197训练采样。每半batch按startup<16/later、各自sequence均匀取样。
这里的采样按可用phase序列均匀；不声称与旧joint逐个draw相同。两臂逐draw完全相同。
不拟合future8/fingers/forecast，故与旧shared比较只能诊断拟合，不能归因为单一学习率。

固定6000步终点评估，不选DEV最优点。复用execution-gap相同TRAIN256/DEV256和四条
序列面板，另终点全TRAIN/DEV，分startup/later。XYZ是未clip意图native命令经过controller
scale的PD误差；Euler报MAE。保存1/1000/…/6000权重与optimizer，不覆盖旧产物。

Decision Note：A准确则优先原joint优化/接口；B显著优于A（DEV均值相对>=20%，startup
也改善）则当前teacher信息值得接回策略；仅TRAIN精确DEV差则优先覆盖/泛化。两者差
则不足以区分信息不足和本次拟合不足，不重复盲目延长。正向离线信号仍需闭环测试。
GPU6/7并行、仅Shadow4/5继续，剩余四卡空闲。每臂<=900s、<=1GB，总<=2GB，保留
20GiB磁盘。预算/非有限/设备冲突只停止本轮，不改现有作业。单seed Probe，无成功率claim。

产物：outputs/ptv3-shared-wam-v2/current-wrist-mlp-20261015-r1/{state_goal,plus_current}/。
脚本：tools/run/probe_current_wrist.py。执行提交由各manifest记录。
