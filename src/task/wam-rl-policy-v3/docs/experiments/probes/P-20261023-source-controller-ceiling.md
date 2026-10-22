---
schema: ref2dex.probe.v2
probe_id: P-20261023-source-controller-ceiling
experiment_id: P-20261023-source-controller-ceiling
date: 2026-10-23
task: wam-rl-policy-v3
branch: main
git_commit: recorded-in-run-manifest
claim_id: C1
hypothesis_family: HF-source-controller-ceiling
probe_index_in_family: 1
seed_pool: probe
seeds: [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139]
decision_changed_if_positive: qualify source-only controller on the same forty development initials before comparing WAM
decision_changed_if_negative: retain all failures and repair measured approach or closure without relaxing physical gates
status: RUNNING
run_id: source-ceiling-20261023-r4
---

## 结果解读表

| source40结果 | 结论 | 下一步决定 |
|---|---|---|
| ≥32/40且物理限位显著减少 | 本20物体开发面板的source控制资格达到目标 | 才讨论同初态/后段WAM比较，不归因数据量单因素 |
| 较14/40提升但<32/40 | 修复有局部作用，天花板仍未达门 | 保留完整40分母，按剩余死因继续修，不跑WAM |
| ≤14/40或限位未减少 | 该版卸载/余量/限速未解决控制瓶颈 | 保留负结果，修控制，不松gate或重训v6 |

分类Blocker：没有可执行source基线时，WAM差距无法归因。最便宜合规方法是复用已有原始轨迹定位控制缺口，再按机主n≥40要求只跑source一臂；0训练更新。

**冻结面板**：原v11同20物体×2不同序列=40回合，不是每物体40回合。native seed100–139、model seed180–219、初始化196、env1/30Hz/600控制/20秒全部照旧；R仅在原source表达拒绝时作为原有无τ后备，保留且单列。全程不实例化WAM。
对象/序列/原始参考/worker归属从v11 registration逐项复制，只将order收窄为A0；40个初态九字段与旧A0对照，容差2e-6，不换样本、不成功即停。此面板已暴露，今后称开发面板，不再称接口未见。

**固定预算**：最多24000控制，0updates，不做RL/BC/v6训练、不做完整RELEASE；每worker≤14400秒、排队≤14400秒、仅空闲GPU4/5/6（≤64MiB），失败保留partial，不自动重采。原生Host、TaskController、S2c阶段/尺寸目标阈值、真实关节/table gate、抓后RelativePoseServo均从v11源码快照继承。

**修复目标（尚非实测结果）**：成功14/40→≥32/40；接近/闭合实际关节中止12/40→≤2/40；40个初态对齐；0物理gate放宽；0训练更新。若未达标继续修source，不比较WAM。

实现为独立source_ceiling模块，保留并行v13默认控制器；12N通用指尖压力阈值、1.2rad/s手指目标限速、候选目标40mrad内余量/根位置10mm余量、0.30m/s和1.2rad/s接近限速。过载/实际关节向限位滑行时退让，并暂停source参考游标；力无效不触发力卸载。阈值未经物理批次验证，明确不宣称正确最优。抓后原伺服逐字节对照测试通过。

代码/复现/逐物体证据：[本轮修复报告](../../audit/source-ceiling-fix-review-20261023.md)。
产物`outputs/wam-rl-policy-v3/source-ceiling-20261023-r3/`；旧source40实测为14/40、16745控制、12限位回合/13关节事件。

预启动修订：原r1已冻结但未获GPU；查出close→lift手指目标跳变与最终目标日志失真，分别由红测试复现。r1 supervisor明确记CANCELLED_PRESTART、0子进程/0控制，原快照和日志保留；物理批次改为fresh r3，不混算。修后对12个真实首限位前状态单步重算：12/12目标准入、6/12触发卸载；这不是反事实仿真，不声称已防止任何旧失败。

## R4运行前修订：实测游标与有界起动（R3不停止、不改评分）

R3已完成至少13个失败回合，因此40回合至多27成功，已不可能达到32门槛；仍完整执行40，不拿部分n冒充整批。
实测trace表明远离物体时游标已走到离开阶段；加上object-anchored闭合后等待运动耦合，缺少有界起动。
按“达不到继续修”执行fresh r4：同一40任务/种子/观察/物理gate/阶段/终点/抓后伺服，再给≤24000控制，0updates；R3+R4累计≤48000，不混算版本。
R4仅当R3至少9个已完成失败、数学上不可能达到32时才可启动；R3仍完成原预算。GPU仍仅空闲4/5/6。

修复：source游标比较当前/下一录制**状态**与实测腕位姿在物体坐标内的误差，仅更接近下一帧才推进；每tick最多1帧，受压时暂停。
39/40参考有有效双指力接触，接近参考最多到该帧+0.5秒；p034无此参考证据，保留样本和missing标记，不伪造接触/不更改任务。
实测既有grasp_candidate成立且尚未运动耦合时，最多10mm/0.06m/s起动，固定原点不累计；失去有效接触撤回起动，成功仍需真实运动耦合+5cm抬升+1s+原尺寸目标。
预期：远离当前参考状态时游标推进1→0；接触候选起动0→≤10mm，不假造grasp；最终成功≥32/40，限位≤2/40。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r4 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r3
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r4 --detached
```
