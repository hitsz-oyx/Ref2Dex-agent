# 中断续跑、三处修复、覆盖与残差试验交付

## 结果解读表

| 结果 | 结论 | 下一步决定 | 来源 |
|---|---|---|---|
| seen达到32/40且干净A2显著较低，但原未见面板不通过 | seen工程接口可保留，不能声称泛化或Cm收益 | 保留冻结R/WAM；按新增初态的真实正控定位接近问题 | Hermes的n40门槛；外部审计扩覆盖 |
| 新覆盖source正控成功而R链失败 | 对该初态有可执行见证；接口/意图是瓶颈 | 训练漂移恢复意图，维持原安全gate | 外部审计区分接口与任务；Hermes修接口 |
| 全链失败或source准入失败 | 正控/执行/任务资格未解决，不能归因WAM无效或任务不可能 | 保留逐例误差、关节和接触记录；不追加本轮训练 | 外部审计；尚无两方独立复跑一致性 |
| BC训练误差下降但DEV或物理不改善 | 当前标签/小预算只提供离线或可运行性证据 | 不扩预算，不恢复三臂RL；DAgger后置 | Hermes的BC重定位及有条件PPO |

v10已完整收齐并通过原始轨迹复核；新几何覆盖36回合与raw复核已完成；BC/PPO与前后两初态试跑已全部完成并复核；本执行流停止追加训练。
所有结论只限合成+8cm位置HOLD，不做完整RELEASE资格。

## v10 全量结果与决定

| 子集/臂 | 严格成功/40 | Wilson95 | 末态持握 | 关节/桌面穿透中止 |
|---|---:|---|---:|---:|
| seen A1 |35/40|[.738879,.945405]|35|1 / 0|
| seen A2 |0/40|[0,.087622]|0|0 / 0|
| 原未见面板 A1 |3/40|[.025836,.198642]|3|0 / 17|
| 原未见面板 A2 |0/40|[0,.087622]|0|5 / 7|

seen A1对A2 Fisher **2.2728876e-17**，CI不重叠；配对补充McNemar **5.8207661e-11**。
达到Hermes seen≥32/40门槛，保留原R/WAM及消费修复。原未见面板Fisher **.2405063**，
未过显著性/CI联合门，不建立稳健τ收益。原未见A1限位40→0，但17次桌面穿透不能被隐去。

互斥死因：seen=35成功+3接触未抬升+1曾持握后失握+1限位；原未见=3成功+17桌面穿透+
14接触未抬升+3未接触+3抬升后连续持握不足。末态持握但未到位均0；这不同于v9 seen的
16/40持握未到位。新v10未见仍以接近/接触物理问题为主，不能因限位减少便转为方法资格。

全部终态误差中位（含提前中止）seen A1 **.006414mm**、原未见A1 **571.232mm**；
原未见A1完整20s生存者仅23，误差中位 **1141.613mm**，须独立看待，不能把中止终点当共同20s。
这些是无真实传感器噪声的仿真状态数值，不是现实机械精度保证。

80/80九字段配对初态一致；80/80 A2 raw检查τ最大值=0、24帧抓姿内容误差=0、R调用保留且
输出非零；160/160 raw独立重评分一致，标签翻转负控制均被拒绝，PD最大重建误差
**3.26348e-7**。common-state非τ字段/valid字节一致另有单元检查，不要求干预后轨迹状态相等。

实际有效回合 **81466 controls** + 中断 **881** = **82347 controls**；保守carry账83266同时保留。
0训练更新。权威结果：`outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/summary-final.json`，
旁有`initial-parity-final.json`、`ablation-final.json`、`g4/g5/g6-final-rescore.json`和
`v10-death-geometry.json`。不合并v9分母、不重标15mm、不宣称40独立物体或预训练贡献。


## Hermes 直接重复的确切入口

在仓库根目录执行以下命令，输出必须fresh；启动前看`nvidia-smi`，只使用空闲4/5/6。
独立session关闭stdin、日志落盘；脱离对话后仍运行。`supervisor.json`保存展开后的全部确切
worker命令和每卡输出。GPU冲突/工程错误停止，不能自动换seed补样本。

```bash
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
TOOLS=src/task/wam-rl-policy-v3/tools/run

# 完整独立重复 v10：80初态×A1/A2，160回合，每臂每子集40，600 controls/20s。
# 严格沿用v10冻结consumer/pose/stage/host；新几何实现不会混进去。
$PY $TOOLS/run_grasp_suite.py --mode v10 \
 --output outputs/wam-rl-policy-v3/grasp-v10-stat40-hermes-repeat1 --detached
# 完成后：该目录 summary.json / supervisor.json / g4,g5,g6/manifest.json。
# 期望完整计数160、无重复，实际成功数/CI由物理结果给出，禁止预填40/40。

# 独立重复新几何合同覆盖：12初态×3链，36回合，≤21600controls。
$PY $TOOLS/run_grasp_suite.py --mode coverage \
 --output outputs/wam-rl-policy-v3/grasp-v11-coverage-hermes-repeat1 --detached
# 完成后：该目录 summary.json；单seed覆盖诊断，不是n40正式泛化结论。

# 本轮中断续跑的实际命令（已启动，不要再次调用相同输出/attempt）：
$PY $TOOLS/continue_grasp_batch.py \
 --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1 \
 --source-suffix resume --attempt resume2 --detached
# 精确worker均为run_grasp_repair_worker.py，--worker 0/1/2，--gpu 4/5/6，
# --registration outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1/registration.json，
# --output .../g4-resume2、.../g5-resume2、.../g6-resume2，
# --carry-run .../g4-resume2-carry、.../g5-resume2-carry、.../g6-resume2-carry。

# 查看本轮实际续跑进度，不再读取旧RUNNING的39回合：
$PY $TOOLS/grasp_repair_progress.py \
 --run outputs/wam-rl-policy-v3/grasp-v10-stat40-20261022-r1
```

模型/统计：`/home2/wyy/miniconda3/envs/graspenv/bin/python`，Python3.8.20、Torch2.4.1+cu121。
native由worker调用`/home2/wyy/oyx_ws/.runtime_envs/dexplore_v120_train/bin/python`，
Python3.8.10、Torch2.0.1+cu118/IsaacGym。R为既有scheduled686、WAM为既有step11000；
不更新这两个checkpoint。输入绝对路径、源码快照、初态、PD、force及阶段均保存在逐worker产物。

## 本轮固定边界与复核

完成39回合复用；三个未完成回合只重置同assigned初态/seed。中断native回执精确记录
161+510+210=**881**controls，原始文件保留。续跑carry保守预留3×600=1800，最终summary同时
保留`environment_controls_conservative_upper`并按native回执回算实际controls；不能把保守上界
冒充实际消耗，也不把中断当物理失败。正式有效批次上界96000，含技术开销实际总上界96881。

后续执行器确切启动命令：
```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/wam-rl-policy-v3/tools/run/finish_grasp_handoff.py \
 --output outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1 --detached
```
这是本轮一次性续作入口，读取现有v10与冻结coverage注册；不是任意重复入口。它先完成
v10 summary、80对初态、A2全80raw消融和三个worker raw重评分，再执行coverage36回合，
然后BC400updates/300s与两回合残差PPO；训练前后各两回合评估，总≤3600controls。
每一步的展开argv、输出路径、日志均在该目录`manifest.json.commands`；工程失败保存并停止。

三处修复的逐项diff、命令、期望数值、参考重放与边界见
[几何/SE3/覆盖报告](grasp-geometry-coverage-20261022.md)。BC/PPO的解释表、标签、reward、
每坐标残差上限、终止与bootstrap合同和预算见
[残差实验卡](experiments/probes/P-20261022-drift-intent-residual.md)。

新增几何合同与旧15mm并报，不重标v9/v10。新增12初态只是便宜的Decision覆盖，不宣称
满足每臂每子集n40的正式效果资格；并发复核主张覆盖也需n40，本简报未给这项额外预算，
此项仍不一致，后续正式覆盖须另冻结n40协议，不能用本轮36回合顶替。

实现预检：原始参考腕/掌间距117.565869/0.109218mm；新两指几何下原参考hold2.000s。
最终交付回归270 passed/0 failed（v3全tests与v6 test_staged_progress，19.43s）；9/9独立源码故障注入被拒绝，均非collection错误。
这些只证明指定合同反例被检测，不把测试PASS当新控制器或几何teacher的物理资格。

Hermes提供本轮修复与条件启动要求；外部审计提供扩覆盖、接口/任务区分要求。独立执行流
曾指出名义PD合法仍可实际关节超界，此边界保留；实测物体滑移不应被固定腕目标忽略的反例
已红绿修复。两方尚未独立复跑本交付，不宣称三方一致。三臂RL继续HOLD。

## 新覆盖与BC当前证据

新几何覆盖36/36完成，raw36/36、三臂初态12/12一致，16046controls、0updates。
新对象A1/A2/A0=2/8、0/8、2/8；新序列为0/4、0/4、2/4。
三条source-only精确成功（airplane_fly、cup_pass、cylindermedium_pass）为局部可执行见证；
A1独有duck成功说明source不是普遍上界。不能把这组单seed面板当n40资格或WAM重训依据。
详见[几何报告](grasp-geometry-coverage-20261022.md)。

漂移恢复BC完成400updates/1.18186s，GPU4约345–347MiB，采样利用率1–14%；小模型和195状态
决定其负载很短，不为追求利用率扩大训练。train MSE **.422871→.0910735**，但DEV
**.386376→.460369**（更差），不晋升为泛化恢复器，不隐去负结果或改选早期checkpoint。
两回合PPO仅完成原注册的有界技术pilot，不以此扩预算；DAgger仍后置。

## 残差pilot最终结果与停止决定

BC的DEV MSE变差，未建立漂移恢复泛化；确定性before **0/2**、after **0/2**。
PPO随机训练0/2，**826controls、2次optimizer updates**；三阶段合计2669/3600controls，
BC400updates另列。所有回合都没有接近/接触/抬升/持握事件阳性；不把这个小预算负结果
升级为v6或RL方法否决。after关节中止1→0也不是抓取收益，当前不追加训练。

三阶段2/2初态一致、raw6/6一致、PPO reward全部五分项重放2/2一致，policy logprob/GAE/
残差界审计PASS。更新后真实Gaussian KL .611887/.165042>target .03，说明仅停止后续epoch
不能保证首次更新的硬KL界；作为本配方限制保留，不隐藏或临时改参补跑。
额外policy重放因允许GPU均被注册作业占用而显式CPU执行，原因/环境写入审计。

[残差卡](experiments/probes/P-20261022-drift-intent-residual.md)有BC/PPO确切复跑命令、
参数与输出路径。结果在`outputs/wam-rl-policy-v3/drift-intent-residual-20261022-r1/summary-final.json`，
邻有before-train-after-initial-parity.json、三个raw重评分与ppo-policy-audit.json。

独立并发流程的[正式n40覆盖](experiments/probes/P-20261022-grasp-coverage40.md)已接续GPU4/5/6，
仍在运行，未并入本报告已完成的36回合/学习pilot，也不提前填其成功率。关于覆盖数量的分歧
通过独立protocol分流处理；三方物理独立复跑的一致性、几何/力代理充分性仍未裁决。
三处修改已实现且在新物理run实际消费，但新控制器没有整体资格，不据此覆盖v10的旧15mm结果。
不重训v6、不做完整RELEASE资格，三臂RL仍HOLD；DAgger后置。

最终门禁：`VERIFY_BASE=2b1da246 python3 tools/verify.py --changed`为PASS，日志
`outputs/wam-rl-policy-v3/grasp-v11-preflight-20261022-r1/handoff-final-verify.log`。
完整测试日志为同目录`delivery-tests.log`；269/265是较早测例集合，保留各自历史日志，
不以旧passed数覆盖并发新增测例或失败。最终270全通过；所有输出仅在项目内。
