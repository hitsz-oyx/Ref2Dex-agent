---
schema: ref2dex.probe.v2
probe_id: P-20261015-full-teacher-input
experiment_id: P-20261015-full-teacher-input
date: 2026-10-15
task: ptv3-shared-wam-v2
branch: main
git_commit: 440f9c13
claim_id: C1
hypothesis_family: HF-full-teacher-input
probe_index_in_family: 1
seed_pool: probe
seeds: [180, 181, 162]
decision_changed_if_positive: replace teacher future reference blocks incrementally with tau after qualifying the supervised full-input mapping
decision_changed_if_negative: diagnose fitting and same-execution labels before attributing failure to WAM or tau
status: UNCLEAR
run_id: full-teacher-input-20261015-r1
---

# Full-teacher-input supervised upper bound

Class: Decision. User explicitly requests actual DExplore1442D delta1+delta16
without tau, supervised on the same q_PD labels. Serves Mission autonomous
execution by distinguishing information mismatch from fitting/data problems.
This is an offline diagnostic, not a learned grasp success claim.

## Data and engineering gates

Use the exact native-current660 rollout, retained TRAIN110250/dev12049 windows,
original sequence split and observed-fixed-grid admission; no new simulation,
no joining contacts from other executions. Retain previous native-current
TRAIN/dev512 panels (seed162, half startup), plus source sequence/tick identity.
Only one current control q_PD[t] is predicted, unlike previous8-step Chunk-R.
Never directly compare new versus old loss as a single-factor information test.

The source stored delta0 only but also saved fullbody state, forces, current
reference428 at every tick. Reconstruct delta1+delta16 by compiling the two
original pure tensor observation functions; preserve measured current IG from
saveddelta0, recompute reference IG differences for each future reference.
Reference clamp is min(t+delta,T-1), matching same-capture terminal repeat-last.
Use pinned Torch2.0.1; metadata admission is exported separately from graspenv
because the pinned simulator runtime does not include spconv. No simulator is
created. Cache stored in full-teacher-cache-20261015-r1, raw source unchanged.

Fail closed per sequence: reconstructed delta0 all721 maxabs<=1e-4; frozen
original teacher (original ReLU MLP, external RMS epsilon1e-5/clamp[-5,5],
clamped deterministic mu) reconstructs same-tick applied action maxabs<=2e-4
and wristXYZ PD error max<=0.2mm. Teacher weights are used ONLY for this audit,
never to initialize or train any student. Capture called get_action(obs,True),
so this dataset's teacher actions are deterministic, not sampled policy noise.
PD-label/action/q same-tick reconstruction was already independently audited.

## Four bounded arms

All students are random-initialized MLP1442→1024→1024→1024→512→12, ReLU, no tau.
Same seed180, matched drawseed181, AdamW3e-4, decay1e-4, batch512,
clipgrad1, standardized single-step active12 PD L1, TRAIN-only input/output
statistics, scale floor1e-3. Current721 is padded with721 zeros to preserve
exact capacity; corresponding normalization leaves masked values zero.

- full_absolute: actual1442→absolute ACTIVE12 q_PD, literal user upper bound.
- full_relative: actual1442→q_PD minus currentq for first6, absolute independent
  fingers; fixed decoder adds measured currentq. q is not a network input.
- current_relative: captured delta0 721, same relativePD head and decode.
- memory_full_relative: same1442 relative model, only fixed TRAIN panel's unique
  windows, 4000 updates; tests fitting on a small subset, not generalization.

The relative arm is necessary because teacher input contains local geometry/
reference errors without a guaranteed world absolute root coordinate. Native
teacher itself computes q_PD=currentq+controller(action) for first6. Failure of
absolute but success of relative must not be mistaken for poor data or WAM.

Three main arms each12000 updates, one budget; evaluate TRAIN/dev512 every500,
select dev wristPD XYZ p95 then mean; memory selects training p95 then mean.
Report selected/final mean/p95, raw PD and executable clamped PD errors,
rotation/finger p95, clipping, curves and selected predictions/normalizers.
Matching observation-precision teacher replay is an engineering gate only.

Mapping fit is locally PROMISING if full_relative dev mean<=20mm; information
mismatch is only locally PROMISING if also >=50% lower than matched
current_relative mean (p95/tails reported separately). If both fit well, do not
attribute old Chunk-R failure to missing future information alone. If memory
fits but dev does not, prioritize generalization/distribution; if memory also
fails, optimization or model fitting remains unresolved. No formal causal or
information-theoretic claim follows from a single-seed Probe.

## Resources and stops

GPU0 cache reconstruction<=1200s, then GPU0/1/2/3 four students in parallel,
three main arms<=1200s each and memory<=1200s/4000 updates; <=4 simultaneous
GPUs, global cap6. Cache<=2GB, each arm<=768MB, aggregate<=6GB;
filesystem>=20GiB. Stop on nonfinite, teacher replay failure, foreign assigned
GPU consumer, budget/time violation. Source/checkpoints unchanged; no official
student initialization and no WAM inference. Record actual code, PIDs,
utilization, memory, elapsed and ETA in each output manifest/progress.

## Engineering smoke

Original metadata loader in pinned runtime failed before computation due to
missing spconv; split CPU admission export from pure native observation cache.
Two train+two dev sequences passed: delta0 max2.38e-6, teacher action max5.66e-7,
PDXYZ max0.000224mm. This demonstrates correct reconstruction for these sequences
only; full corpus must pass per-sequence before scientific fitting.

## Limitations / future evidence

Repeated dev use, one seed, coarse source quality gate, no physical execution.
A1442 win over the old tau model also changes architecture/output horizon;
only matched721vs1442 relativePD isolates the bundled future reference change.
It does not identify t+1 versus t+16 contributions or contact specifically.
Substitution of teacher reference blocks with tau and physical deployment are
follow-ups only after interpreting this upper bound; not included here.

## 首轮结果后的最小预处理判别（运行前记录）

全660条teacher replay通过，delta0 max2.38e-6、teacher action max2.89e-6、
XYZ PD max0.000795mm。首轮12k已完成：full_relative selected dev45.70mm/p95134.35，
current_relative49.77/149.58，absolute126.33/308.06；memory TRAIN3.16/7.69mm。
这些是单步直接PD的新模型，不是旧8步R分数的可控替换。首轮full_relative训练误差
仍明显高于memory，dev与TRAIN也有差距，不能据此确认输入充分但监督无效。

Class: Decision。最便宜剩余检查为同随机初始化/数据/标签/12k预算的两臂
full_relative与current_relative，改用teacher原RMS epsilon1e-5和[-5,5]截断，
排除TRAIN z-score无截断与teacher真实预处理不一致的优化影响。current仍721后补零，
对应有效721维取teacher首分支RMS；归一化后第二分支强制0，不引入参考信息。
teacher仅提供固定预处理统计，仍无官方网络权重初始化。它是额外特权预处理诊断，
不是与首轮单因素input对照；本轮input因果对照仍在每个相同预处理内部进行。
GPU0/1并行，各12k、1200秒、768MB；与首轮合计仍在6GB/6卡资源边界内。
完成后停止预算，不追加无界模型/优化器搜索；保留首轮全部结果。

## 最终结果与独立审查

Result: COMPLETED，UNCLEAR。首轮4臂、teacher-RMS补充2臂均正常完成、GPU已释放。
主运行代码f1b7026c，补充代码16d964e4。缓存重建38.21秒，全660序列共122299个
准入监督tick通过teacher replay（不是全部165677个原始动作）：action max2.89e-6、
PDXYZ最大0.000795mm。原始同次观测、label时钟及deterministic teacher mapping成立；
这个工程证据不等同源rollout是成功抓取示范。

| Input / target | Selected step | TRAIN mean mm | dev mean mm | dev p95 mm |
|---|---:|---:|---:|---:|
| 1442 absolute PD | 3000 | 62.69 | 126.33 | 308.06 |
| 1442 relative PD | 4000 | 22.78 | 45.70 | 134.35 |
| 721 relative PD | 4000 | 26.88 | 49.77 | 149.58 |
| 1442 memory (TRAIN only) | 4000 | 3.16 | 88.75 | 299.49 |
| 1442 relative PD + teacher RMS | 11000 | 14.46 | 42.80 | 135.03 |
| 721 relative PD + teacher RMS | 9500 | 19.51 | 48.97 | 144.02 |

表中按预定dev p95优先选点；memory单独按训练集选点，其dev仅为说明不能泛化。
首轮最终12k全量TRAIN/dev mean：full_relative14.11/44.01mm，current17.26/50.05mm。
teacher RMS selected full14.46/42.80mm，对照19.51/48.97mm，mean改善12.60%，
p95改善6.24%；TRAIN-only归一化mean改善8.18%，p95改善10.18%。均没有达到20mm
目标，也没有出现相对721的>=50%改善，不确认主要问题就是缺teacher未来输入。

memory使用509个唯一TRAIN样本，mean3.16mm/p957.70mm；说明小集合可以拟合，
不证明全量函数已经学会或dev泛化合格。literal absolutePD为126.33mm，结合relative
显著更好、原生控制需加live q，不能将此失败解释成teacher label/时钟必然错误。

独立只读review：重算6臂selected XYZ mean/p95与选择步数一致；TRAIN589/dev71
序列不交叠，固定TRAIN512/dev512面板逐项等同旧native-current实验，均256startup+
256later；full/current同4107788参数、step0权重逐tensor一致、relative目标统计相同。
所有学生fromscratch、无tau、无官方网络初始化；RMS仅为固定预处理。source admitted
q_PD/action/q同tick已审计，teacher重放进一步减弱了时钟错位/采样随机性解释。

Decision: 停止本轮拟合预算，不立即替换r[t+1]/r[t+16]为tau或继续WAM微调。
完整teacher输入在当前学生训练下仍存在独立于WAM/tau的拟合/跨序列泛化瓶颈。
接下来应先围绕全量TRAIN14mm与dev43mm差距、稀疏/异常状态和学习目标定位；
数据示范质量风险仍在，但不能直接归因于坏示范。当前单seed未完成优化/容量/质量
的matched因果区分，不把UNCLEAR升级成information mismatch被否定的结论。

产物：`outputs/ptv3-shared-wam-v2/full-teacher-input-20261015-r1/summary/`保存
results.json/csv、table.md、curves.png、independent-metric-audit.json与manifest.json。
首轮/补充每臂保留curve/result/manifest、selected checkpoint和对应预测；cache有
逐序列teacher replay审计、source/split/tick、固定面板和数组。输出可追溯，不覆盖原
660数据、微调权重或既有实验。

## 资源记录更正（后续任务核对）

根STATE显示九源远端四卡已于14:07启动；上一轮首批本地4卡学生短拟合与之重叠，
实际超过全局6卡上限，原资源段只写本地4卡是不完整的。四臂已全部结束，不影响
其他任务的进程或权重；不追溯称为临时7卡例外。后续teacher-fit-generalization对照
已改为仅本地2卡分批，与远端4卡合计6卡。root在继续任务时已向用户说明并纠正。
