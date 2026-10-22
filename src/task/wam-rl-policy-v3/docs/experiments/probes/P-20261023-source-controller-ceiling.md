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
run_id: source-ceiling-20261023-r17
---

## 结果解读表

| source40结果 | 结论 | 下一步决定 |
|---|---|---|
| ≥32/40且物理限位显著减少 | 本20物体开发面板的source控制资格达到目标 | 才讨论同初态/后段WAM比较，不归因数据量单因素 |
| 较14/40提升但<32/40 | 修复有局部作用，天花板仍未达门 | 保留完整40分母，按剩余死因继续修，不跑WAM |
| ≤14/40或限位未减少 | 该版卸载/余量/限速未解决控制瓶颈 | 保留负结果，修控制，不松gate或重训v6 |

分类Blocker：没有可执行source基线时，WAM差距无法归因。最便宜合规方法是复用已有原始轨迹定位控制缺口，再按机主n≥40要求只跑source一臂；0训练更新。

**冻结面板**：原v11同20物体×2不同序列=40回合，不是每物体40回合。native seed100–139、model seed180–219、初始化196、env1/30Hz/600控制/20秒全部照旧；原v11中R仅在source表达拒绝时后备；R5起source-key/motion各版均不消费R，forward仅诊断并单列，不能再称原source-PD原样重放。全程不实例化WAM。
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

## R9预注册：接触处停止插入，探测不打断闭合

仅当R8至少9个完整失败、32/40已不可能时启动；R7/R8均完整跑完，不修改其快照/分母。R9同40/≤24000控制/0updates，前七个物理版本R3–R9累计上限168000控制；不比WAM，不放宽评分或gate。
R7已有21个回合发生抓姿锁定，但锁的是未到达的目标，距实测腕中位63.232mm、最大178.322mm，接触后仍可继续推物。R9锁定首次匹配接触时的**实测腕**，闭合期间不再插入；持续0.12秒失触/卸载仍恢复接近。预期该红测试腕推进10mm/tick→0；不是把接触当成功。
R8的接触候选探测又过早替换闭合目标：尚未耦合时用弱力持握微调冻结了半张开的手。R9保留原source闭合命令间1.2rad/s限速及12N卸载；探测只改最多10mm腕起动，不提前覆盖手指闭合。预期q=.4时命令.4→.44，而非停在.4附近；仍须真实运动耦合/抬升5cm/持续持握1s/原尺寸终点误差才成功。
两条实际step红测试分别见R8/contact-stop-red.log、probe-closure-red.log；预期32专项/28mutants。物理成功目标仍32/40、预抓限位≤2，未知能否达到。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r9 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r8
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r9 --detached
```

## R9预启动取消 / R10预注册：分离目标内余量与实测危险

R9为CANCELLED_PRESTART、0子进程/0控制；其注册、源码和取消记录保留。实际step红测试证明普通接触形变q=.01（名义下限0、目标内余量.04）即触发父类换抓姿，即使仍远未达到物理abort−.10；source模块虽没有卸载，父类仍偷偷重规划。R10在保留R9两处闭合修复的基础上，增加**可选admission.measured_pressure**接口，只有source MarginAdmission使用名义边界外.05且向外运动的实测预警；普通控制器默认行为不改。候选PD目标仍严格在原名义范围内留40mrad余量，真实physical abort仍±.10。
冻结器在v11 baseline Geometry上仅注入两行调用钩子，保留原postservo/评分；不把当前并行控制器整份替换进去。预期上述普通形变的换抓姿次数1→0，但真正危险仍卸载并受原gate约束。33专项、29mutants、全v3为309 passed/0 failed/0 skipped（20.85s，2warnings），不当物理成功。
R10同40/≤24000控制/0updates，前序是R8（R9未启动），R3–R8六版加本版最多168000控制。R7已完整0/40、22073控制、0jointabort；R8至少30完整失败，但仍跑完40，不截分母。新目标仍≥32/40/预抓限位≤2，不达到不比WAM。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r10 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r8
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r10 --detached
```

## R10技术中断 / R11预注册：腕命令限速与manifest竞态

R8已完整0/40、23387控制、0jointabort。R10在3/40完成（0成功）时因读取R8正在重写的空manifest触发JSONDecodeError，supervisor终止自身两个worker；GPU4组未开始。R10实际2494控制：1624属于3个完整回合，870属于2个中断回合，35回合未开始；保留全部partial，不把它写0/40或计算完整n40比较。该错误不涉及他人进程；两客户端finally关闭各自native。
R11修race：worker JSON用临时文件原子替换；reader对旧writer的空/截断manifest按pending处理，不释放卡、不终止正在运行的其他本批worker；queue超时仍有效。三项CPU队列回归含“空文件当COMPLETED”和“直接覆盖原manifest”故意错误。
同时修腕限速真实缺陷：旧代码把“每秒.75m”实现为每tick相对**实测腕**永久截断PD偏移.025m，而非相邻**命令**的位移限速，削掉native PD跟踪/运动前馈。新命令以先前最终PD目标作SE3限速参考，阻挡q不动时六拍目标.025→.150m、每拍仍≤.025；实测卸载/最大10mm候选探测仍保留实测安全锚点。原抓后伺服/评分/物理gate不变。
同40初态/24000控制/0updates/20秒，直接以前序完整R8为队列前提（不能等永远FAILED的R10）。R3–R8实际138782，加R10中断2494=141276控制，本版后≤165276，未突破此前168000的总上限。预期source≥32/40/预抓限位≤2；CPU通过不代表达到。红测试R10/wrist-slew-red.log实际.025 vs期望.150，非缺参数失败。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_source_ceiling.py src/task/wam-rl-policy-v3/tests/test_source_ceiling_queue.py --basetemp=tmp/pytest-source-independent-final
$PY src/task/wam-rl-policy-v3/tools/audit/run_source_ceiling_mutants.py
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r11 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r8
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r11 --detached
```

## R12运行前修订：闭合完成/历史清空后才做耦合探测

R11至少19个完整失败，32/40已不可达，但仍完成40，不换分母。已出现真实完整成功，不能选例称达标；全量数字以本版summary为准。最初9个完整回合中，失败例p002/p005/p004首次probe时距原计划closure还差0.229/0.269/0.366rad，成功例剩余约0.04rad（含inner projection），表明探测仍可能早于闭合。不是统计归因实验，而是已由实际step红测试复现的相位接口缺口。
R12把候选force+原反馈条件保留，并要求**内余量投影后的闭合命令完成**、随后持续原InteractionConfig.history_s=0.25秒，再起动同一个≤10mm探测。这个等待是为清除闭合手指自身运动占据的相对运动历史，不伪造q/dq/接触，不修改评分/后段/物理gate；已由真实运动建立的grasp不等探测。
预期首次probe时指令余量0.23–0.37rad→≤1e-6rad，完成后候选持续≥.25s；实际关节是否稳定仍待物理trace，不拿命令完成当持握。CPU定向38项（35控制+3queue）、32/32控制mutants通过，queue另抓2种注入错误。目标source≥32/40、预抓限位≤2，尚未证实。
按“达不到继续修”追加同40初态/≤24000控制/0updates/20秒；前序R11。R3–R8及R10中断已141276，加R11/R12各≤24000，累计新上限189276（明确追加本版24k，不冒称仍168k）。GPU仅等空闲4/5/6，旧批不中止；不比WAM，不RL/不重训v6。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r12 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r11
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r12 --detached
```

## R13运行前修订：恢复原source-PD，仅增加最小压力/余量防护

R11完整4/40=10%（Wilson95 3.958–23.052%），23080控制，1jointabort/1关节事件；相对原14/40显著退步，Fisher=.0144757444。R12已有≥36失败、32门不可达，仍完成40。此前keypose/时钟/单点停腕/probe重写均未改善原source，不继续把重写后的链称原source-PD。
R13采用单独SourcePDGuard，恢复原v11的source-action→18维PD target→strict source_pd_action(live q)→原GraspIntent/MeasuredGraspConsumer阶段/RelativePoseServo路径；不静态换keypose、不新增停腕、不新增probe、不把PD误差冒称速度。只有：①实测>12N/原名义±.05向外危险时退让/0.1秒脉冲并暂停参考；②候选/目标留原内余量；③目标准入与实际危险分开，原physical abort±.10和评分不动。安全源intent在无危险/不需投影时与原控制器命令逐位相同，postservo也逐位相同。
source reference每个pregrasp控制前进1帧，卸载暂停，held后段不再耗掉参考；与初态同一世界native q，不声称object-anchored GTτ/R上界。原无τ R后备仅在strict source逆变换拒绝时选择，完整报告selected/实际pregrasp-consumed/拒绝原因；不能把有R辅助的成绩谎称纯source-PD。source-key提取只作旧元数据兼容，明确不执行；WAM不实例化。
预期原健康source动作差异→0、postservo差异→0；原12/40限位→≤2，主目标14/40→≥32。这不是已达标；39控制+3queue专项通过，当前Guard独立4个mutants覆盖改写健康输入/忽略压力/改后段/漏内余量，旧32个legacy-controller mutants仍保留，不能互借测试结论。
同40/24k/0updates/20秒，前序R12。累计已有R3–R8+R10+R11=164356控制，加R12≤24k和本版≤24k，新累计上限212356，明确追加本版，不隐瞒改进失败成本。若仍<32，继续按原source剩余死因修，不比WAM。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r13 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r12
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r13 --detached
```

## R14运行前修订：指令带宽/跟踪偏移准入，与参考推进配套

R12完整2/40，23795控制/1jointabort；R13完整12/40=30%（Wilson18.075–45.430%），14279控制/19jointabort/20关节事件；相对原14/40 Fisher=.81165，不能称提升。R13恢复原输入后表现回升，但预接触冲击仍不受控。19个限位回合中，末次PD目标变化的中位数为位置317.064mm、native腕关节范数0.1991rad、主动指关节最大0.1398rad；最大为761.977mm/2.7867rad/0.5647rad（其中紧急卸载也可能造成目标回撤，不能全归为源指令跳变）。
R14只在pregrasp正常控制加：相邻PD指令主动手指≤1.2rad/s、native腕joint范数≤2.4rad/s、位置远/近≤.75/.30m/s；与实测q的PD跟踪偏移远处≤75mm/0.15rad，近处位置≤既有geometry.wrist_gap，角偏移按该尺寸与手跨度限定。**跟踪偏移是柔顺/压力代理，不冒称实际速度或接触力保证**；卸载优先可例外，实际native gate不改。后段命令仍原样。
参考不能照墙钟跑：手指带宽未达/卸载/最终投影例外时暂停；根目标受限且没有原生产多指contact_observed也暂停；已实测多指接触允许reference继续给后续手形，避免因不可达接触PD预载死锁。held阶段冻结source参考。三种pending/clock注错在CPU队列测试覆盖，未伪造task事件或成功。
预期正常指令最大跳变降到≤25/10mm、.08rad腕joint、.04rad手指；阻挡PD偏移受上界；source帧在受限无接触时每拍1→0，而非继续越过接触参考。目标关节中止19/40→≤2，成功12/40（原14/40）→≥32，未知是否达到。健康对照fixture明确是**同时通过带宽/余量**的输入，仍逐位同原，不能泛称原14回合完全不变。
同40/24k/0updates/20秒，前序R13。所有已完成/中断合计202430控制，本版后≤226430（明确追加24k）；不比WAM，不RL、不重训。复现红见R13/bandwidth-red.log。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r14 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r13
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r14 --detached
```

R14最后启动检查还抓住“未执行的handover closure也阻塞reference”：执行preshape未受限时不得因未来hold initializer更大而冻结cursor；handover本身仍≤.04rad初始化，不让持握开始瞬间跳到完整closure。正常带宽不足、柔顺/native投影优先、紧急卸载三种口径分别记，不拿0canonical clipping冒充无干预。软件完整计数见R14/tests-ready-all.log；当前production Guard的8项直接step回归、11个故意错误与旧legacy模式检查分开，不用legacy PASS替Guard物理正确性背书。

## R15运行前修订：实测接近区允许手形继续，不等先有接触

R14已≥17完整失败，32门不可达但仍跑满。p000满600控制停在reference36（source首次接触42），t180–599 root受限、无多指接触、0卸载、手已近物体：同一个实测接近区开启柔顺限制，clock却要求先contact才继续，闭合手形永远送不下来。R15只改reference pacing：使用同一InteractionConfig.approach_distance_m=.06的实测finger gap，近物体即可跨过root受限继续给后续手形；手指带宽/卸载/最终投影仍阻塞。没有将接近当接触，更没有改lift/hold/成功。
production Guard动作算法保持R14；身份改为稳定schema `source-PD-guard.force-margin-bandwidth.v1`，reference协议为`source-PD-guard.measured-approach-paced.n40.v1`，与run_id r15/实验ID分离，不再共用连续版本号。旧冻结产物标签不改。
预期实测gap40mm、contact=false、root受限时reference36→37，远离仍36；第一条回归的旧语义实际返回36（AssertionError，非缺参数失败），见R14/approach-clock-red.log。实际手指/对象是否建立接触与运动耦合仍须物理验收。目标仍source≥32/40、jointabort≤2。
同40/24k/0updates/20秒，前序R14；累计上限明确追加24k至250430（R14实际控制数完成后再报），不比WAM/不RL/不重训，保留R14完整40。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r15 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r14
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r15 --detached
```

## R16运行前修订：源参考绑定live物体，ACQUIRE不越过抓取参考

R15已有≥17完整失败，32不可达但仍完成40。hand案例无live持握却把reference推进到188（source闭合完成52），继续回放GT抓后移动；旧world-q参考亦不随live物体漂移。R16沿同一已测试的native mount/SE3公式，把原PD腕目标经live_object @ inverse(recorded_object)映射，再严格逆回live native q；手指PD目标不变。参考只推进到产生source闭合完成状态的前驱command_frame，之后继续真实物理控制直到live耦合进入原后段。完整录制/TaskSpec/标签不删不改，绝不把source接触当live成功。
新增生产helper接口测试：live对象平移50mm，腕目标应0→50mm；对象旋转.4rad，实际native mount下腕姿态同旋转，手指不变；live acquisition最多到command_frame，不能走入已录制postgrasp/release，原12动作/13状态仍完整保存。旧生产world/full-horizon语义3项实际断言失败，见R15/reference-contract-red.log，不是缺函数或import错误。三种故意回退各被测试抓住。
另修“从held失持的第一拍”仍沿旧phase绕过Guard：红测试实际移动目标400mm/预期≤10mm；现在按实测失持先进入恢复保护，正常held后段不改。见R16/recovery-command-red.log；不声称这解释全部掉落。Guard schema v1不与实验run编号混用。
目标source≥32/40、pregrasp jointabort≤2；测试不是物理通过。source direct Guard新增9个step回归/12mutants，legacy兼容35/32分列；reference helper3项带3种注错，queue5项带4种注错。
同40/24k/0updates/20秒，前序R15；累计预算明确追加24k，上限274430（结算报实际）。不比WAM、不RL、不重训v6。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY -m pytest -q src/task/wam-rl-policy-v3/tests/test_source_pd_reference.py src/task/wam-rl-policy-v3/tests/test_source_ceiling.py src/task/wam-rl-policy-v3/tests/test_source_ceiling_queue.py --basetemp=tmp/pytest-source-reference-independent
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r16 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r15
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r16 --detached
```

## R17运行前修订：按实测腕/object几何及离桌阶段选参考

R16未达32，仍完整40。hand/camera失败末态无触力，live腕相对对应source实测腕仍差93.7/109.0mm；追上PD motion-feedforward终点不等于抵达真实抓姿。R17不改Guard/S2c成功判据，用实测object-local腕SE3及object-table gap匹配记录状态，只有当前离桌阶段允许的source状态可选，再取最多2个记录帧（原30Hz中66.7ms）的前瞻action。参考允许随漂移回退，GT frame不冒充真实elapsed时间；finger命令未就绪/卸载仍不向前跳。完整源数据保持原状。
参考phase上界用live table gap+既有geometry.position_tolerance_m，拒绝source object已抬升而live仍在桌面的过早姿态；无phase候选时显式phase_admission_missing并选最近gap参考，不删该回合、不宣称候选有效。匹配转角乘实测手点到腕的杠杆长度，全部误差为m，无物体/初态ID常数。
同时控制器入口只允许“原measured coupling +明确unsupported”才进入lift；桌面推着object移动不等于可持握。仅复制controller feedback把established压低，绝不改原TaskController/评分的force、contact或成功，不把False升True，正常held后段不改。源轨迹的真实抬升动作提供起动，不新增假tau/强制物体运动。
预期live supported时选择的source gap不越过现有尺寸容差（无候选例外明报），接近参考不再按墙钟推进到错误抓姿；supported/unknown时controller grasp-established=0。几何平移/旋转共变、phase边界、supported推物三种故意错误均被回归抓住。目标source≥32/40及pregrasp jointabort≤2仍待物理。
同40/24k/0updates/20秒，前序R16，累计上限明确追加24k至298430；不比WAM/不RL/不重训。源码/测试不把PASS当物理通过。
```bash
export TMPDIR="$PWD/tmp"
PY=/home2/wyy/miniconda3/envs/graspenv/bin/python
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r17 --prepare --predecessor outputs/wam-rl-policy-v3/source-ceiling-20261023-r16
$PY src/task/wam-rl-policy-v3/tools/run/run_source_ceiling40.py --run outputs/wam-rl-policy-v3/source-ceiling-20261023-r17 --detached
```


## 06:30完整结果与资源中断点

R14/R15/R16完整分别0/40、5/40、4/40，控制24000/23408/23440；各自关节中止0/2/2。R16 Wilson[3.958%,23.052%]、相对原14/40 Fisher双侧p=.0144757444，明确退步；32/40未达。全部逐物体/q/dq/PD/阶段/几何见各run summary.json，报告头部汇总，未删除失败。累计已确认273278控制（含R10 partial）；R17新≤24000，总≤297278。

R17冻结提交64356b6c，全量331 passed/0 failed；冻结reference入口6 passed/0 failed。supervisor pid331584处于QUEUED，0 worker/0控制；GPU4/5/6分别被外部会话训练占用约22.6/23.0/19.4GiB。只等空闲，不杀进程、不转用7或0–3；4小时队列预算，超时明确FAILED并保留现场。自动完成后按原40评分/raw重算/初态比对出summary.json，未达32仍不比WAM。
