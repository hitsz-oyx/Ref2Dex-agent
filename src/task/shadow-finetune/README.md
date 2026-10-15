# ShadowHand 原生动作微调与抓取闭环

用户要求暂不做Inspire迁移，直接用已有UniDexGrasp++状态—动作执行训练WAM-v2+Shadow动作头，
并在原ShadowHand仿真环境报告成功率。Task属于WAM到独立执行底座的控制资格验证。

当前是小规模工程/Decision Probe，不把小样本或单次成功当正式结论。完整数据量效应须在
同一Shadow环境内比较固定小子集与全量，固定模型/优化曝光预算/heldout对象和reset种子；
Shadow成功不单独证明Inspire失败由数据量造成。

- 父模型：WAM-v2 remote80k，保持原hand/scene forecast分支，新初始化300D current state+shared query actor。
- 动作：8×24 canonical原生60Hz命令，根6维force/torque+18维finger控制；每60Hz只执行首拍重规划。
- WAM：H4间隔1/30s、future24共0.8s，对应原始60Hz历史[-6,-4,-2,0]与未来[2,4,…,48]。
- 任务输入：当前已知目标位置，保留当前物体姿态构造goal delta；不用未来实测物体作为输入。
- 当前300D观测包含官方当前力/传感/上一拍动作/64D静态geometry feature，是本轮明确的state profile。
- Shadow raw landmark已经按palm/五指根/五指尖排列，不套用Inspire的索引置换；另一手invalid。
- TRAIN-only state/action mean/std；actor loss对归一化原生命令L1，保留hand/scene forecast；无GTτ warmup。

数据从已有raw重导，不修改其原始文件。pilot先选原20物体中的8TRAIN/2DEV候选，
保留末帧物体距任务目标<=5cm的轨迹，屏蔽物/手速度>5m/s或转速>40rad/s后的全部窗口。
这只是BC候选，不等价稳定抓持认证。当前实际7TRAIN物体41条/6232窗、2DEV物体10条/1520窗。
一个被选TRAIN物体没有合格候选，记录后不制造样本。完整候选cache现已从2066物体来源准备：
筛选后TRAIN1567物体/10118轨迹/1536897窗、DEV388物体/2338轨迹/354761窗，物体交集0。
尚未进行全量GPU微调。

入口：

```bash
python tools/run/prepare_data.py --output <fresh outputs/shadow-finetune/cache> --pilot-only --train-objects 8 --dev-objects 2
python tools/run/make_eval_panel.py --cache <cache> --output <fresh panel.json>
python tools/run/launch_probe.py --data <cache> --panel <panel.json> --output <fresh run> --gpu 6 --steps 1000 --batch-size 64 --detach
```

使用graspenv Python3.8/Torch2.4.1；IsaacGym始终先于torch导入。native评估复用已验证的固定collector，
仅在独立runtime副本替换actor调用；不改vendor，完整保存动态补丁和raw动作。teacher/student同seed，
初始化RNG围绕模型构造保存/恢复，再对q/object/hand初始状态逐项数值比较，失败则拒绝配对结论。
同时记录官方sticky5cm、末帧5cm、连续60个60Hz观测在5cm内且抬升>=10cm三种结果；
后者是保持位置代理，不宣称已证明接触稳定或现实成功。小面板2TRAIN+2DEV、每物体4个起点。

用户临时允许额外一张空卡做小规模测试：原6卡Inspire继续、Shadow用GPU6、GPU7空闲；
这是本轮临时授权，不自动扩大后续长训预算。CPU准备30min/16GB，GPU pipeline最多1h/14GB，
20GiB空闲底线。父权重/其他训练、Viser均独立。详见Task实验卡。

首轮1000步pilot已完成，按固定DEV action L1选择step500。16env原生闭环teacher官方success5/16，
学生1/16；分开看学生TRAIN1/8、DEV0/8，teacher分别3/8、2/8。同步抬升保持代理学生1/16，
初始状态完全一致。这仅确认小规模监督和执行链路，数据量假设仍为UNCLEAR；GPU6已释放。

全量cache的12456条已逐条CPU审计通过：动作上一拍时钟max误差0、raw/CAD身份匹配、
安全tick支持集合一致、TRAIN/DEV物体无交集。可用
`tools/audit/audit_cache.py --cache <cache> --output <fresh audit directory>`复核；
报告在`outputs/shadow-finetune/full-cache-audit-20261015-r1/report.json`。

下一步大小数据对照用`tools/run/prepare_scale_comparison.py`生成共享controls与命令，
两臂均以small TRAIN拟合state/action统计、使用同full DEV64窗及同父模型/初始化/预算。
`train.py --controls <controls.json>`会拒绝settings、输入manifest或DEV物理身份漂移。
准备入口不启动GPU；详见[数据量对照卡](docs/experiments/probes/P-20261015-shadow-data-scale.md)。
