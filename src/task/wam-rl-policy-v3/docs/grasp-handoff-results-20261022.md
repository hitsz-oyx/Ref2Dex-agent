# 中断续跑、三处修复、覆盖与残差试验交付

## 结果解读表

| 结果 | 结论 | 下一步决定 | 来源 |
|---|---|---|---|
| seen达到32/40且干净A2显著较低，但原未见面板不通过 | seen工程接口可保留，不能声称泛化或Cm收益 | 保留冻结R/WAM；按新增初态的真实正控定位接近问题 | Hermes的n40门槛；外部审计扩覆盖 |
| 新覆盖source正控成功而R链失败 | 对该初态有可执行见证；接口/意图是瓶颈 | 训练漂移恢复意图，维持原安全gate | 外部审计区分接口与任务；Hermes修接口 |
| 全链失败或source准入失败 | 正控/执行/任务资格未解决，不能归因WAM无效或任务不可能 | 保留逐例误差、关节和接触记录；不追加本轮训练 | 外部审计；尚无两方独立复跑一致性 |
| BC离线改善而PPO/物理不改善 | 当前标签/小预算只提供离线或可运行性证据 | 不扩预算，不恢复三臂RL；DAgger后置 | Hermes的BC重定位及有条件PPO |

当前：v10续跑中；新几何/覆盖/BC/PPO均已预注册，顺序执行器等待v10收齐与raw复核。
本页将回填实际结果，不能把中途比例当n40结果。所有结论只限合成+8cm位置HOLD，不做完整RELEASE资格。

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
最新完整回归265 passed/0 failed；9/9独立源码故障注入被拒绝，均非collection错误。
这些只证明指定合同反例被检测，不把测试PASS当新控制器或几何teacher的物理资格。

Hermes提供本轮修复与条件启动要求；外部审计提供扩覆盖、接口/任务区分要求。独立执行流
曾指出名义PD合法仍可实际关节超界，此边界保留；实测物体滑移不应被固定腕目标忽略的反例
已红绿修复。两方尚未独立复跑本交付，不宣称三方一致。三臂RL继续HOLD。
