# 新采刚体 pilot：来源绑定、准入与覆盖（2026-10-23）

## 结果解读表

| 观测 | 含义 | 决定 |
|---|---|---|
| 96/96 条完整准入；26,304/26,304 窗口准入；goal 96/96 可认证 | 新producer从reset前指令到reset恢复、语义与窗口合同已连通 | 具备按机制扩采条件；下一批建议N=384，成本见下；本轮不启动 |
| 本轮不可认证/拒绝数=0/0 | 未用缺证数据冲准入率 | 所有数据仍保存在独立候选目录，未混入生产TRAIN |
| 原108格中零格 61→51，减少10格（16.4%） | 仅真实source4机制格获得新增覆盖 | 将全量合法数据的固定面板对照重训列为后续任务；其余来源仍缺，不冒充全域补齐 |
| 固定面板预测变化=NOT_RUN，训练updates=0 | 尚不知道补覆盖是否改善预测 | 不宣称预测收益；未来覆盖补上而同面板不动时停止买数据，转条件/标签审计 |

96条×10s@30Hz，28,800输出ticks；每条另存初态边界，共301状态。
保持原生60Hz actor/physics，每2controls记录一次，合计57,600原生controls；
reset warmup另96controls。没有重训、模型替换、checkpoint、视频或删除历史产物。
原6k manifest仍`5bda6a99678b0586b496eff7b4b3ba3afddda996cdc16c43603341ed1264225f`，
原condition recipe/tuple/seed170与固定512面板未改。旧goal认证仍0/6,000，本次只给新执行认证。
新增索引独立，不自动接入TRAIN；`candidate-manifest.json`将每个tuple的sequence_ordinal
明确绑定到`rollout-verdicts.jsonl`行序，再到实际rollout/registry/reset指令。
该ordinal不是plan.jobs顺序，也不是旧6k sequence ID，完整26,304行已精确核对。

实验`P-20261023-rigid-source-bound-pilot`，run`rigid-source-bound-pilot-20261023-r2`，
branch main；初始采集代码480ae4e4，批次恢复入口8d97deae，后续每批实际代码在
`batch-*/simulation-config.json`，总execution/resume receipt和collector快照保留。
[实验卡](../experiments/probes/P-20261023-rigid-source-bound-pilot.md)；
[机器摘要](rigid-source-bound-pilot-20261023-summary.json)。

## 从源头写对的五个字段与实际验收

每条具有`reset-binding.json`、`goal-channel/instruction.json`和实测`raw.npz`。
指令生成于独立constructor角色，并在第一次reset_target_pose调用之前写入batch的
`intent-before-reset.json`；source_role明确为explicit_task_instruction，记录实际同clock
available/reset纳秒时间、内容SHA、execution/reset/object/hand slot。来源不是自身未来。
现有native reset会在一次初始化中调用两次reset_target_pose；两次均恢复冻结的goal/init/root
状态并保留单一逻辑reset及重复调用receipt，物理采集中出现reset则拒绝拼接。

独立落盘重读通过现有`classify`来源分类器与`validate_instruction`，逐条核对源hash、
指令派生、时间顺序、原生goal持续相同、reset恢复。任务目标是reset声明的task_completion
pose，无到达deadline；局部scheduler waypoint保持独立诊断，不替换任务目标。

手的21点来自同一执行的named rigid bodies；现有ShadowSurface从q22/root13重建并与live21
逐帧交叉核对，最大差2.60025263e-06m。该source的wrist约定仍是palm原点，
不是解剖腕点；拇指CMC/MCP/IP与tip有显式映射。778表面点、16part、2048物体表面与32query
均有明确固定ID/坐标/米/法线与资产绑定。左手不存在，mask为false，未伪造第二只手。
接触沿原v5 3mm/10mm nearest-surface-vertex proximity，event沿原K24迟滞/persistence合同；
不把proximity称为实测force或指令性接触，不强填unknown尾部。

全部26,304个H4/K24候选逐窗运行现有admission_reasons，实测通过
sample_window并由新显式`reset_bound_sample`替换goal三个字段，在线ResetWaypointChannel逐位一致。
旧sample_window默认自身t+24 endpoint继续只作旧合同标签，禁止用于新数据部署goal。
未来手/物标签干预测试验证其不能改变新goal；其他6k读入入口未接此adapter。

## 配额与实际覆盖

source4旧格中10个为0。前期文字曾误写9，完整矩阵驱动的配额计算未受影响。
冻结配额由旧phase×PCA几何窗口数的水位缺口分配，
无物体身份阈值；flat37/elongated23/compact36。先在本地已有官方TRAIN资产包中，
排除官方TEST seen/unseen母资产，按固定archive顺序取每个数值几何类前两项，
同格按固定轮转分配。刚体statue等只按仿真刚性与数值形状归类，不作生物/功能类别推断。
approach延迟、contact的统一开指时间表仅按机制施加；完整plan/seed/命令全部保留。
预定phase实际出现90/96条，不把未达成的配额当成标签。

| 实际准入且goal认证窗口 / geometry | flat | elongated | compact |
|---|---:|---:|---:|
| approach | 772 | 850 | 584 |
| close | 713 | 653 | 592 |
| contact | 368 | 493 | 350 |
| hold | 8213 | 4266 | 8270 |
| unknown | 72 | 40 | 68 |

原108格完整前后CSV保存在run目录，所有source列保持原身份；仅source4叠加，
其他8源数字逐格不变。数量是重叠窗口，96条rollout也有共享资产/actor，
不是26,304个独立样本。原512锚点不重抽。

## 实测资源与下一批成本

规划仍报告16–32KiB/tick约0.44–0.88GiB，含registry/日志≤2GiB。
本轮实际run序列化535,070,374bytes，加只读复用的准备资产和失败r1
6,666,512bytes，总0.505GiB。
这是文件逻辑字节，不冒充文件系统元数据的精确分配量。
项目tmp的gymtorch编译扩展另约3.4MiB；Isaac Gym沿既有默认目录自动缓存VHACD，
这是运行库cache，不作为数据/registry或认证证据；不删除已有cache。

写前实测143.14GiB，资源采样最低余量
131.05GiB，报告时余量
116.59GiB，均高于30GiB。
写入前逐次guard，保留并发全量转换；无历史数据删除。
前36条GPU4、后60条GPU5；每批启动前检查空闲，始终只派一张采集卡。
资源偏差：GPU4首两批之间发生启动竞态，preflight空闲后其他执行器也启动，
第二批有短时并存；采样GPU总占用峰值4670MiB包含对方显存，
不能当作pilot独占峰值。发现占用后在批次边界停止新启动并等待，未干预其他进程。
转换释放GPU7后迁到空闲GPU5，保留GPU6/7两张空闲卡。
总采集墙钟1074.7s包含等待；
六批物理循环和raw保存合计82.9s，
不含初始化/import/VHACD或GPU排队。audit墙钟849.2s包含等后续批次，
不能把这些含等待的墙钟当成GPU计算小时。

下一批建议 **N=384**（仅source4，按更新后的phase/几何缺口重新分配，不按对象特例）。
115,200输出ticks、230,400原生controls+384warmup；按本次实际run线性约
1.99GiB，保守沿旧估算为1.76–3.52GiB，含registry/日志
建议4GiB硬上限并保留≥30GiB。建议采集30min、CPU标签/审计20min独立上限，
GPU排队另列；这是基于pilot的规划预算，未执行，不能作为实测吞吐承诺。
扩采只能增加source4密度/新几何，不能补其他8源的来源格；在证明固定面板预测收益前，
不因合同通过而无界买数据。后续全量重训须保持原模型/训练预算和原512面板，另建新增数据身份，
报告overall/contact/first8；比较还须保持相同部署条件策略。若同时启用新合法goal，
必须另列条件对照，不能把收益只归因数据量；未来event仍只作离线标签，
部署event须unknown或另有独立任务来源，goal认证不替event指令发证。本轮按要求0updates。

## 复现、检查与局限

主入口[collect_rigid_pilot.py](../../tools/run/collect_rigid_pilot.py)，
独立落盘审计[audit_fresh_rigid_pilot.py](../../tools/audit/audit_fresh_rigid_pilot.py)，
候选tuple绑定[publish_fresh_rigid_manifest.py](../../tools/audit/publish_fresh_rigid_manifest.py)，
显式goal消费者[fresh_rigid.py](../../src/ptv3_shared_wam_v6/fresh_rigid.py)。
graspenv：Python3.8.20、Torch2.4.1+cu121、Isaac Gym GPU PhysX；每批记录实际环境，
Isaac先于Torch导入，TMPDIR/TORCH_EXTENSIONS_DIR均在项目tmp。
6个batch使用seed198–203，几何198；属于Probe暴露seed，不作正式Validation。
run内plan/launch/processes/execution/resume/simulation-config保存确切命令与版本。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 TMPDIR="$PWD/tmp/rigid-pilot-20261023" \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/run/collect_rigid_pilot.py audit \
 --output outputs/ptv3-shared-wam-v6/rigid-source-bound-pilot-20261023-r3
# fresh prepare/collect后用新run审计；已发布目录拒绝覆盖。
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/audit_fresh_rigid_pilot.py \
 --check-only --output outputs/ptv3-shared-wam-v6/rigid-source-bound-pilot-20261023-r2
# 独立检查可只读复跑；不覆盖已发布receipt。
```

定向合同29 passed；独立重算96条所有候选窗口、来源/时序/reset、每帧surface/part重建、
每条3帧contact及全量event重算和完整前后矩阵PASS。`tools/verify.py --changed`仍因另一Task四张历史卡
30个既有outputs失效链接失败，未伪称全库通过；本轮schema/链接无新增错误。
r1在首次warmup物理步前因初始化重复reset守卫停止，0条raw、0采集controls，产物保留；
r2前两批后遇其他GPU4作业，保留36条已完成batch并恢复，未补采/替换失败样本来抬高准入率。
合同通过只说明记录/标签/条件来源有效，不证明成功抓取、部署收益或跨来源泛化。
