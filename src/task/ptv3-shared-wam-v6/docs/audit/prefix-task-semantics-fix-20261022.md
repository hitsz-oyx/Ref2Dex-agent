# 安全prefix与完整任务声明分离（2026-10-22，CPU）

已修复生产侧标注/读取合同，并将历史501个TRAIN、engineering-PASS、非空prefix任务写到
**新版本目录**。没有删样本、改变旧sidecars或用新成功数覆盖旧负结果；未启动训练、GPU或仿真。

新目录：`outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r1/`。
其`manifest.json`保留501个相同sequence/source及56,418条可用prefix控制；原660条总语料未改，
本次只版本化被历史PPO实际准入的501条TRAIN cohort，不声称重做了其余159条。

## 根因与最小修复

冻结3a0779f6的
`tmp/inspire-measured-official-20261020-r3/frozen/src/task/wam-rl-policy-v3/tools/audit/audit_teacher_rollout_corpus.py:120–121`
先取`clean[:cut+1]`，再重新推断终态和事件，只用`replace(source_length=n)`恢复总长度。
因此保住了source_length数值，却已经换掉任务终态、终点和未来事件。
同冻结树`src/wam_rl_policy_v3/teacher_dataset.py:30`进一步要求任务终点等于prefix边界。
这把“安全数据截止”变成了另一个较短任务，并非原完整任务的删失观测。

修复接口：

```python
declared_task, availability = task_with_safe_prefix(
    raw_task, safe_prefix_commands, unsafe_future=...
)
```

- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/task_annotations.py:17`的新helper返回同一个
  完整TaskSpec；终态、完整object_path/source_frames、source_length、原CONTACT/RELEASE
  source_frame均不重推断。`task-availability.json`单独保存控制可用边界与删失信息。
- `src/task/wam-rl-policy-v3/tools/audit/audit_teacher_rollout_corpus.py:118`调用此helper，
  不再调用截断数据的`task_from_teacher_archive(clean)`；即使prefix为空，也不把原任务改写成UNKNOWN。
  `:54–55`拒绝复用旧prefix语义sidecar，避免通过symlink把旧错误带回生产。
- `src/task/wam-rl-policy-v3/src/wam_rl_policy_v3/teacher_dataset.py:34`要求task.json与存在的
  raw-task.json完整声明一致，旧截断task明确报错并要求迁移；`:39`独立检查prefix availability，
  `:46`核完整source_length。读取时仍只产生安全prefix可用动作，允许完整任务终点位于其后。
- 原始未来事件仍放在完整task.json和availability.declared_events中；超出可观测prefix的事件
  标为删失，**不写NONE、不作为负事件标签**。训练样本也不补造超prefix控制。
- `complete_positive_eligible=false`、`terminal_intent_verified=false`：删失/unsafe future不能
  充作完整任务成功正例；未删失也不自动认证物理成功或计划意图。

新reader保留旧的“未截断完整任务”fixture兼容性；有raw-task却与task不一致的历史prefix侧文件
被拒绝。生产准入需要显式切到新manifest，不能继续拒绝旧64个UNKNOWN来代替语义修复。

## 501条预期与实际交叉表

预期来自逐份旧raw-task.json：RELEASE490、HOLD11。实际修复结果相同；旧UNKNOWN64条全部保留，
不是删除后让分母变小，也没有把它们统一改HOLD。

| 旧prefix模式 | 新完整RELEASE | 新完整HOLD | 合计 |
|---|---:|---:|---:|
| HOLD |408|8|416|
| RELEASE |20|1|21|
| UNKNOWN |62|2|64|
| 合计 |490|11|501|

终态恢复改变**473/501**（408+1+62+2），不变28；不是469。
491/501标为删失且有unsafe future；709个超prefix的原事件保留。
完整验证结果`integrity.json`：

| 验证 | 实际/预期 |
|---|---:|
| 相同sequence/source/prefix cohort |501/501|
| admission.npz字节完全相同 |501/501|
| raw-task.json字节完全相同 |501/501|
| legacy-prefix-task.json与旧task.json字节相同 |501/501|
| 新task.json完整packet等于原raw-task |501/501|
| source_length保留 |501/501|
| 原event(kind,source_frame)对保留 |1221/1221|
| 原有序source_frames保留 |81794/81794|
| 可用prefix commands |56418/56418|
| 完整成功正例自动认证 |0/501|

旧数据原样保留；新版本还复制`legacy-prefix-task.json`便于逐例审计。
该版本的任务/goal/event条件不同于旧prefix任务，**不能凭shape或相同动作行数复用旧task10/τ cache**。
没有重新训练BC/PPO，也没有将新声明回填成旧训练已经使用的条件。

## raw-task也不是天然正确的计划意图

完整raw-task来自实测force+近物体接触区间的启发式推断。接触消失可能是计划释放、失握、推动后
分离或数值故障。此次仅恢复完整历史声明，不把任何失握重标成成功RELEASE，也不把raw490R当作
490个已认证的可执行计划。

审计逐条读取同一真实归档`reference_current`显式二进制CONTACT字段，仅作意图线索对照，
没有作为新质量准入或新训练GT条件：501/501可读，均为reference RELEASE；交叉表raw R→ref R490，
raw H→ref R11。即**11条raw HOLD也不等于原参考要求保持**。存档参考标签与tick对齐可核，
它仍不是独立人工意图/机器人执行成功证书。本次不擅自把11H改成R。

490个raw RELEASE中：

| 指标 | 样本量/实际值 |
|---|---|
| 最后RELEASE早于完整raw路径终点 |490/490|
| RELEASE至终点时间差(s) |min0.2333 / median2.0 / mean2.0743 / p90 2.7367 / max15.0333|
| 释放后物体中心累计路程(m) |median0.020794 / mean0.186012 / p90 0.146742 / max4.176625|
| 释放至终点净位移(m) |median0.010009 / mean0.105081 / max2.521417|
| raw最后释放与某存档reference释放相差≤3frame |333/490|
| 释放后掉落代理：之前coupled，之后下降≥5cm且unsupported/uncoupled |37/490|
| 上述代理且没有±3frame reference释放时点见证 |29/490|

掉落代理不是因果滑落分类；可能仍包含有意释放后的运动、飞行或数值失败。
不能把其余453条认证为成功，也不能用29直接删除样本。所有原任务保留，事件未重标。
字段及阈值见`repair_prefix_task_semantics.py:31`的audit_release与每条release_audit。

## 阶段时序合同的具体证据（未在本修复中改scheduler）

| sequence | raw RELEASE / reference RELEASE | 完整终点 | 释放后中心路程 | 下降≥5cm代理 |
|---|---|---|---:|---|
| s6_bowl_drink_1 |171 /170|240（再2.3s）|99.159mm|否|
| s2_hammer_use_2 |233 /232|306（再2.4333s）|5.715mm|否|

这两例存档参考的释放时点与raw差1帧，不能一概解释为意外失握。
旧TaskScheduler要求完整路径/终点先到达并完成0.2s settle才进入RELEASE，因而可能要求抓持
越过原始release后的路径段；之前的对齐trace中bowl173、hammer235已回到ACQUIRE。
这是需要单独修阶段时序/成功gate的证据，**本次只修任务声明与数据边界，不改scheduler或终点**。
未将490条统一截到release pose，也没有删除post-release路径。

## Red → green 与可复核命令

遵循已授权的两个公共边界：标注器输出完整task+prefix metadata；TeacherRolloutCorpus读取
完整task但只提供可用控制。没有mock内部scheduler来制造成功。

- `prefix-task-red.log`：修前7 failed。其中真实旧reader拒绝完整task+短prefix；反而接受手工
  复现的“截断task+恢复source_length”错误，测试为`DID NOT RAISE`。
- helper第一slice为5 passed；完整prefix/既有task_annotations/teacher_goals回归**13 passed**。
- 显式故意错误：重新运行旧clean[:prefix+1]推断会得到UNKNOWN/HOLD、丢失未来RELEASE，必须与
  完整声明比较失败；reader也拒绝此错误packet。缺失未来事件不能靠填NONE让测试通过。
- 六种旧→新mode cell各选一条真实reader smoke均PASS，完整cohort仍501；其控制长度分别为
  52/49/81/177/83/8，与完整source_length212/159/209/178/264/178分离。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python \
  src/task/wam-rl-policy-v3/tools/audit/repair_prefix_task_semantics.py \
  --output outputs/wam-rl-policy-v3/inspire-task-prefix-semantics-20261022-r2

CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q -p no:cacheprovider \
  src/task/wam-rl-policy-v3/tests/test_prefix_task_semantics.py \
  src/task/wam-rl-policy-v3/tests/test_task_annotations.py \
  src/task/wam-rl-policy-v3/tests/test_teacher_goals.py
```

r1已存在，重现迁移必须用新目录；命令中的r2仅为复核示例，未运行。只刷新既有统计可用
`repair_prefix_task_semantics.py --refresh-audit`，不重写task侧文件。

当前源码相对冻结3a0779f6的逐行diff在新目录`prefix-semantics-code.diff`，完整本轮实现快照在
`code/`；未执行git操作或改历史冻结代码。可用`diff -u`对照以下三对文件，差异退出码1属预期：

- 冻结`.../src/wam_rl_policy_v3/task_annotations.py` ↔ 当前同Task的`src/wam_rl_policy_v3/task_annotations.py`；
- 冻结`.../src/wam_rl_policy_v3/teacher_dataset.py` ↔ 当前同路径；
- 冻结`.../tools/audit/audit_teacher_rollout_corpus.py` ↔ 当前同路径。

机器产物：新目录`manifest.json`、`repair-audit.json`（501逐例含reference/drop代理）、
`integrity.json`、`progress.json`。测试与真实reader日志在
`outputs/ptv3-shared-wam-v6/negative-result-audit-20261022-r1/prefix-task-{red,green,regressions,reader-smoke}.log`。
本修复未授权任何新方法比较；完整任务物理正控制、raw intent与阶段gate仍须分别验收。

## 追加：同501份完整raw task的旧/新评分器离线对照

本节只重放已有机器人档案，不产生动作、不调用物理或PTv3模型；不把它称作新执行成功率。
输入为上述新迁移manifest的同501份完整task，安全prefix仅是同一次因果回放在边界处的snapshot。
旧60ef与新版分别在**独立Python进程**内加载，避免同名模块缓存混用；未运行旧UNKNOWN controls。
旧包来自`outputs/ptv3-shared-wam-v6/teacher-task-scoring-20261022-r1/old/src/task/wam-rl-policy-v3/src/`；
新包完整冻结到`outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/new/src/`，schema为
`ref2dex.measured-task-scheduler.release-at-declared-pose.v2`。实际活跃链逐文件核对，只有
task_scheduler.py不同，task_feedback/interaction_state/goal_scheduler/task_runtime/task_spec均字节相同；
`active-code-diff.patch`保存完整差异。两者包含声明event pose里程碑/释放授权/释放后路径等合同修复，
不是把其中某一行单独消融的因果实验。

每版2个CPU worker，合计4；全501分别耗时155.17s/154.22s，均COMPLETED。

| 相同完整raw任务，观测范围 | n | 完整TaskController成功 | RELEASE授权 | 确认path complete | 录制结束删失 | 明确评分器失败 |
|---|---:|---:|---:|---:|---:|---|
| 安全prefix，旧60ef |501|0|0|0|499|2 path_no_confirmed_milestone|
| 安全prefix，新v2 |501|9|11|10|490|2 path_no_confirmed_milestone|
| 原完整录制，旧60ef |501|0|1|1|492|9 path_no_confirmed_milestone|
| 原完整录制，新v2 |501|77|205|204|415|9 path_no_confirmed_milestone|

新77个完整录制成功全为RELEASE；**只有9个成功在原安全prefix内，68个在其后**。完整原录制包含
既有unsafe/删失后缀，不能把77自动当作合格示教或新teacher正控制。原机器人行为、controller阈值
8mm/.12rad、0.2s路径settle、1s终态窗口均未由本次审计调整；结果只说明旧评分合同确有假阴性影响面。
原来“0/501”必须限定旧评分器身份，不能继续解释成这些档案物理上均无法完成。

先跑的两条定位例在旧版均未授权RELEASE，新版两例都完成：

| archive | 新RELEASE授权 | 新完整成功 | 实际终态窗口 | 旧阻塞 |
|---|---|---|---|---|
| s6_bowl_drink_1 |tick171=5.7s|tick222=7.4s|6.4→7.4s|173失握后退ACQUIRE|
| s2_hammer_use_2 |tick233=7.7667s|tick269=8.9667s|7.9667→8.9667s|235失握后退ACQUIRE|

两例两版的`path.initial_acquired`从tick0均为True，不能归因“最初锚点从未获取”。新评分器在
同一失握反馈前已经按声明释放pose授权RELEASE，继续检验post-release路径及完整终态。
它们都是既有archive的重评分，不与新host上teacher/source-PD的另一批0/6混成同一物理执行。

### 真实终态删失：不补帧、不降低1秒门槛

全录制501/501含各自raw最后source_frame，之后的额外观测时间均为0。几何到达可以比原最后source
时刻更早，所以77例仍确实有≥1s观测窗口；不能把“录到了最后参考帧”直接当成已观察完整终态保持。

新版RELEASE已授权但未成功128例全部删失，互斥分解为：

- 116例path已完成，末尾实际terminal-settle已开始，但记录不足1s；
- 11例path已完成，末尾释放/稳定/支撑条件没有形成有效终态窗口；
- 1例授权RELEASE后路径尚未完成（s6_wineglass_drink_2）。

具体`s8_bowl_drink_2`：tick150授权释放，path complete；tick220=7.3333s开始终态窗口，
录像末tick249=8.3s只有29/30=**0.9667s**，缺下一tick，正确记删失。没有重复末帧或运行新仿真补足它。
未进入RELEASE、未完成路径和未观测足够终态时间是不同缺口，均不自动等于物理无能。

### 仍待裁决的候选几何进度撤回（没有擅自修改）

新版全501的初始锚点均已获取，但76例最终milestone pointer=0。只对以下两个真实案例做了
逐帧机制定位，不将其机制直接推广成全部76例的原因：

| archive | 已观测的候选路径进度 | 单帧接触阈值跌落 | 撤回 | 后来首次motion coupling |
|---|---|---|---|---|
| s10_stamp_pass_1 |tick28：pointer0→1，force3 bodies/near4|tick29：force0/near3，grasp_lost=False|pointer1→0、last_pose清空|tick42=1.4s，首待办源28已距410.157mm（meanCAD411.920mm）|
| s10_binoculars_pass_1 |tick43..84：pointer0→30|tick85：force1/near4，grasp_lost=False|pointer30→0、last_pose清空|tick103=3.4333s，首待办源43已距269.854mm（meanCAD271.498mm）|

两例下一帧force分别恢复2/4 bodies，near均4，但已错过被撤回的早期里程碑；后续档案没有同时满足
“重访该目标pose且execution_ready”的帧。初始anchor仍True，问题是**未确认抓持之前的几何观测
因接触瞬断被撤回**，随后完整raw路径要求回到已经经过的历史点。详见candidate-rollback-diagnostics.json，
其witness含阈值、真实输入和scheduler状态。

这证明候选证据撤回与完整raw路径之间存在合同兼容性问题；是否应该保留独立的几何见证frontier、
将“抓稳/阶段”证据单独要求，需明确裁决。也可能需要给原始接近/推拨/抓持段赋予正确阶段语义。
本审计没有修改该逻辑、降低阈值、补CONTACT标签或把两例直接计成功。

### 追加产物与复核

产物根`outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/`：old/new.json逐501记录，
comparison.json/md配对汇总，old/new-smoke.json含bowl/hammer完整trace，new-diagnostics.json含候选撤回
定位trace，candidate-rollback-diagnostics.json为其独立几何/force核对，new-dwell.json为0.9667s实例。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 /home2/wyy/miniconda3/envs/graspenv/bin/python \
 src/task/ptv3-shared-wam-v6/tools/audit/rescore_prefix_tasks_cpu.py \
 --task-root outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/new \
 --implementation release-at-declared-pose-v2 --workers 2 \
 --output outputs/ptv3-shared-wam-v6/prefix-task-rescore-20261022-r1/new-recheck.json
```

旧版替换task-root为上面的old包所在Task根，另写old-recheck.json。汇总命令用`--compare --old-file ...
--new-file ... --output ...`；候选诊断用`--task-root .../new --diagnose-rollbacks .../new-diagnostics.json
--output .../candidate-rollback-recheck.json`。这些是可复核命令，不表示已额外重跑全量。
