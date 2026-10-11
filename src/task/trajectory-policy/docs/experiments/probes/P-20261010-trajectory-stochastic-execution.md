---
schema: ref2dex.probe.v2
probe_id: P-20261010-trajectory-stochastic-execution
experiment_id: P-20261010-trajectory-stochastic-execution
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: 959cd74408ddf7fbbc6e83e24ed10505838e06fe
claim_id: C3
hypothesis_family: HF-trajectory-policy-deployment
probe_index_in_family: 1
seed_pool: probe
seeds: [294, 295]
decision_changed_if_positive: validate stochastic standalone trajectory policy before WM training intervention
decision_changed_if_negative: inspect actual trust-region update movement before more task training
status: UNCLEAR
run_id: trajectory-stochastic-execution-20261010-r1
---

# Is stochastic deployment hiding a learned usable trajectory policy?

## Motivation / Decision Note

Mission A needs a usable self-trained physical policy; Mission B still needs
matched Cm policy-training benefit. Lambda1 task Probe improves startup credit
but frozen mean still0/4 while GT/dense3/4; occasional sampled training holds
could reflect learned stochastic behavior orchanceexploration. Same-H lambda1
precontact XYZ mean movement<=.03893mm percoordinate versus1mm std. Before
changing optimizer orpayingfor more training, test the actual Gaussian policies
with controlled common noise. This is a **Decision** Probe: robustfinal sampling
gain leads tobaseline verification, absentgain leads to update-movement replay.

Root selects one64env wave with28warm/final pairs andeightcalibration rows,
instead ofonly4pairs, to avoid an uninformative tiny stochastic sample. Estimated
~1--2GPUminutes, no parameter fitting. Changing onlydeployment action selection
for frozen learned rows keeps the H/D/R/native chain. No Mission/claim change,
new branch, force/phase/referencepolicy input oroldTask experiment. Resource
caps below andno repeatwave. Parallel same-reset/common-noise rows are not
exact hidden-physics forks; solver divergence remains a limitation.

## Fixed protocol

Warm actor:startup-balanced-actor-20261010-r1/best.pt; finalactor:onlyfinal24
fromtrajectory-ppo-long-credit-20261010-r1 (0487233). Verify immutable actor,
D/R/native/data identities; no modelupdates, optimizer orcheckpoint selection.
Independent absolute c=mean(H)+std(H)*epsilon, raw Gaussian beforeD bounds;
no addedreference base. H onlyown measured fourpast/currentstates, resetpadding,
24futureplan/8executedprefix/terminal6, fixed542nativecontrols andstrict highest
precision. Same gains/frame0/airplane/reward-free evaluation aspreviousmeanwave.
Four tauGT andfour densecalibration rows are clearly privilegedcontrols only.
Twenty-eight warm and28final learned rows, total64, rolespermuted by seed294.

Generate onePCG64 noisebank with NumPydefault_rng(295).standard_normal shape
(68,28,288), castfloat32 BEFORE model sampling; store noise.npy andSHA before
physics. PairID0:27 assigned bysortedroleindices withinwarm/final, sameepsilon
bank[query,pair] forboth policies. PairID is bookkeeping, neveractorinput.
Actor evaluates its ownH, so pairedmeans/states can diverge; sameepsilon isnot
samec. Useactualtrainedstd withno scalechange. Save every H/mean/std/epsilon/c,
D/FK/queryq/object/velocity, Rinputs/latent/requested/applied/PD andphysicalstates.
Noise isfixed beforeoutcomes, notsearched orselected fromtraining successes.
Seed294 samephysics setting asmeancomparison;295 onlynew Gaussian drawbank.

One complete542control wave. Calibration GT/dense each>=3/4 held>=433 plus
terminalheld. Report everypair: maxima/terminalheld/clipping, final-only successes
versuswarm-only successes, acquisitions anddropout. Baseline usefulness signal
PROMISING iffinal>=21/28 longheld+terminal andclip<1%; otherwiseUNCLEAR. Learning
signal PROMISING ifcalibrated andfinal baselinepasses and>=7morestable episodes
thanwarm (25percentagepoints); UNPROMISING ifcalibrated andbothlearnedzero stable;
otherwiseUNCLEAR. These arefixed exploration screens, not statisticalValidation.
A fewshort orrareheld episodes donot establish useful standalonebaseline.

No extra engineering physics wave: same sampling operation ran duringPPO,
current change addsdeterministic commonnoise andlarger nativebatch. Preflight
checks AST/contracts/resource/inputhashes; actualallquery sampling/native/outcome
replay ismandatory. Independentaudit verifies noisebank/seed/pairedidentity,
mean/std/c onactualH, explicitD nativeEuler/coupling/FK/velocity andR/commands/PD.
Never treat a green wiring check as grasp orpolicy learning.

## Resources / stop

Single idleGPU4, new6GPUmin/256MiB; one64env nativewave<=240s andall audit/
pairedstatistics<=60s. GPU forpolicy/FK/PhysX; CPUfiles/hash/paircounts.
Check preflightGPUandownedprocess, monitor128controls/GPUmemory/utilization/ETA,
foreigncompute guardandfinite/nativecommand equality. Abort timeout/nonfinite/
sourceweightdrift/unsafeGPU/unexpectedtermination; preservefailedmanifestandno
automaticrerun. Alloutputs taskowned/fresh, nocheckpointoverwrites/externalwrites.
Totaloutputs must staybelow300GiB; existing119GiB ample. No other processes touched.

## Results

`959cd74` 一次64env/542control完整冻结wave已完成85.719s，151.51MiB。
GPU4显存约7553MiB，128--384步util37--43%，ETA与预算相符；没有重跑、
训练或模型选择。GT4/4、dense3/4达到>=433连续held且终末持有，校准通过。
warm/final各2/28长时终末持有，medianheld均0，clip分别.9818%/1.0741%。
稳定成功出现在相同pair8/20：warm477/484帧，final481/484帧；未增加成功数。
按固定screen，学习和baseline信号均UNCLEAR，不能称为可用baseline或RL收益。

同提交独立all68query审计通过：H/mean/std/epsilon/rawc/物理解码/FK exact，
velocity3.81e-6，Rfeatures5.72e-6，command2.38e-7，PD0；初始states、noisebank
seed/hash/pairID、实际执行与outcome一致。CPU配对报告工具进一步独立计算
held连续长度、取得/失持、clip并核验同noise/初始H；报告保存到原wave新增的
paired_outcomes.json，既有文件不覆盖。`eb3e96d` 配对统计通过：both2/final-only0/warm-only0/neither26；
warm有6行形成过held，4行随后失持，final为5/3。滑窗/短时held不替代固定433帧标准。

Root据此选择已有数据的**真实Adam状态重建/单update LR恢复诊断**，不是第三次
盲目训练。当前随机策略仍只有罕见抓持，最终策略没有显示稳定成功增量；不关闭
trajectory/WM科学假设，下一卡将先要求逐轮actor/value checkpoint exact，再
比较同batch/同moment的更新尺度。没有新增seed/epochs/物理评价或WM。

Artifacts:
- `outputs/trajectory-policy/trajectory-stochastic-execution-20261010-r1/`
- `outputs/trajectory-policy/trajectory-stochastic-execution-audit-20261010-r1/`


## Limitations / future evidence

Oneinitialstate/motion/seed andonefixednoisebank. Pairing reduces uncontrolled
sampling differences butnot nativehidden-state variance, andonlyactualevaluated
policies aretested. Meanandstochasticdeployment differ; gaussianjitter may
exploit theexecutor without improving a stablemean. No Cm intervention, WM,
pairednative-action training control or formalRL claim. Neednewseed/fullheld/drop
Validation forpositivepolicy signal, then matched Cm training utility.
