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
run_id: source-ceiling-20261023-r8
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

## R5运行前修订：source关键抓姿，而不是录制时钟

R4已有至少13个完整失败，因此同40最多27成功，32门槛不可达；仍完成40。实际trace的PD平衡点并不总是更接近下一录制状态，R4匹配时钟停滞（p005 cursor0满600步）。不再靠更松的时钟阈值绕过它。
R5按v7三项接口直接提取source实际抓姿/预形/闭合目标，物体局部抓姿映射live；抓后目标仍来自同一任务。
39/40参考具有“有效双指力+生产18mm两指邻近”证据，选首次持续≥0.12秒的接触完成状态；p034用最近的双指几何候选，明确无source接触证据，不剔除/不伪造CONTACT。
预形取接触前约0.167秒的实际关节，闭合目标取抓姿状态的实际关节；实测腕到位就允许尝试闭合，即便打开的指尖尚未触物，真实接触/耦合仍是抬升资格。
R的forward仅保留为旧trace兼容诊断，**不消费R输出、不消费source动作/时钟**；这是source-keygrasp控制器，不叫原source-PD原样重放，更不叫GTτ/R上界。
卸载/限位余量/限速/有界起动保留，原抓后伺服/成功/物理gate不变。

fresh R5同40初态、≤24000控制、0updates；仅在R4至少9个已完成失败后启动，R3/R4仍保留完整40。三版累计≤72000控制。
每GPU须等所有前序source版本的该GPU worker完成后才可启动，避免相互抢同一张卡；并行v13不动。
预期：源游标停滞消除（接口不再有游标）；腕到位但开指时approach→close；新source最终≥32/40，预抓限位≤2/40。目标不是实测结论。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r5 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r4
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r5 --detached
```

## R6运行前修订：保留握力预载与闭合完成抓姿

R3/R4/R5均已完整40：各0/40（Wilson95 0–8.7622%），分别23894/24000/23648控制；真实关节中止均0，table中止1/0/6；每版raw40/40一致。比原14/40显著退步（每版Fisher3.07753148e-5），不是改进；不跑WAM。
已消耗71542控制，R6再给同40/≤24000控制、0updates，四版总上限96000（按前三版实际最多95542）。依旧同初态、评分、post servo和物理gate，不换样本。

修正两处真实控制错误：闭合目标取产生所选状态的前一拍source PD命令（action[f-1]+controller），不是接触挤压后的实测q；限速相邻两次**命令**，不是永远绑定实测q。阻挡时目标可0.44→0.8逐步建立预载，不是改实际状态。
source key优先选原轨迹已有“生产observer运动耦合+抬升≥5cm+对象底部离桌>2cm”的闭合完成状态，而不是初碰；40中35有此证据，39有source对象接触，5个不具完成证据的样本仍保留接触/几何候选，不叫live成功。
远离物体时PD位置偏移上限0.01→0.025m/控制（近物仍0.01），对应命令角速率1.2→2.4rad/s；这不是实测速度保证。力槽index/middle/pinky/ring/thumb与native手点thumb/index/middle/ring/pinky显式映射；控制闭合允许匹配的中间指节线段接触，live评分几何不改。
实测卸载预警提前量为原±0.10m/rad中止界限的一半（±0.05），目标仍在URDF内40mrad余量；原actual physical gate不变。名义URDF越界另统计，不能用0中止冒充0机械偏差。

预期：hand thumb pitch闭合目标0.05558→0.45993rad；被阻挡q=.4时12拍命令0.44停滞→0.8且相邻≤.04；首次接触约13s→期望≤7s；最终仍目标≥32/40/预抓限位≤2。CPU测试302 passed/0 failed/0 skipped（19.35s，4warnings）、26项专项、21/21 mutants killed，不当作物理通过。

```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r6 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r5
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r6 --detached
```

## R7运行前修订：闭合接触后固定世界抓姿，实测失触才重新接近

R6完整0/40、21780控制、1关节中止（2关节事件），未达32。原14/40仍较强；不比较WAM。
真实trace与step红测试表明，闭合时继续按live对象映射抓姿会追逐被推动的物体；R7在首次匹配实测指节接触时捕获该世界抓姿，继续向其闭合，不跟随对象滑移。有效实测失触持续0.12秒后释放并重新接近；过载卸载后也重新规划，不冻结陈旧目标。未接触前仍映射live对象，抓后伺服、评分、物理gate完全沿用v11。
此为Blocker修复，决定是否具备source≥32/40资格；最便宜验收仍是同40任务，不加WAM/训练。预注册n40/≤24000新控制/0updates，同种子/初态/20秒，R3–R7累计上限120000控制；本版运行前四版已实耗93322。目标32/40、预抓限位≤2；目标不是实测结论。
软件预期：接触后对象平移50mm，已捕获目标平移0mm；失触0.12s后approach、捕获清除；28专项通过、23故意错误被抓。原生腕XYZ安装变换实测最大漂移1.22e-6rad，排除腕运动学顺序错误。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r7 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r6
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r7 --detached
```

## R8运行前修订：恢复source接近路径与完整腕PD目标

R7已有≥9个完整失败，32门槛不可能达到，但仍跑完整40。静态source key重写的各版均退步，不能继续把它称原source-PD天花板。R8恢复原始source PD路径，用实测object-local腕位姿在前方9个录制状态中找最近单调游标，消费该状态的真实PD动作；不按墙钟前进、不回放参考离开/释放，接近上限为source闭合完成帧。接触后沿用R7世界抓姿锁定。
另一处明确的信息丢失：R6保留手指PD目标却拿实测腕当指令，40条的腕位置PD-minus-measured范数中位67.037mm、p90 153.097mm、最大358.156mm；角差中位0.1142rad。这包含运动前馈/受力偏差，不能称纯接触力。R8闭合腕目标同样来自产生该状态的前驱action[f-1]，映射当前物体，受同一限速/内余量/卸载保护，不放松物理gate。
Blocker目的：恢复被本轮改写丢失的source动作信息，而非为τ归因另加对照。预注册同40初态/24000控制/0updates/20秒；R3–R8累计最多144000控制。source目标≥32/40、预抓限位≤2；未知能否达到。若仍不提升，继续修而不比WAM。新红测试为两项实际输出不一致（不是import/缺参数失败），R7/motion-red.log；软件验收30专项、26故意错误，原抓后伺服/评分/gate不变。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r8 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r7
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r8 --detached
```
