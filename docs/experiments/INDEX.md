# 实验阅读索引

由 tools/experiment_index.py 生成；状态照录原卡，不重判科研结论。
问题、结果与下一步见原卡开头；详细配置和执行版本见其 manifest。

## HB01

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-observation-router-reliability](probes/P-20260926-observation-router-reliability.md) | New-seed reliability of the observation-driven six-expert baseline | 见原卡 | 见原卡 | probes/PROMISING |
| [VAL-20260926-observation-six-expert-c1](validations/VAL-20260926-observation-six-expert-c1.md) | Validation: frozen observation-driven six-expert C1 route | 见原卡 | 见原卡 | validations/COMPLETED |

## HF-action-intervention

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-early-hold-intervention](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-early-hold-intervention.md) | Can actions manipulate interaction information useful for keeping a grasp? | 494 complete early-hold trials; A UNPROMISING, B UNCLEAR for insufficient strata despite 49.6% GT I error improvement; C not executed. | Close this four-step residual contract; retain GT I prognosis, without more seeds, fitting or online selection. | probes/UNCLEAR |
| [P-20261004-randomized-action-intervention](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-randomized-action-intervention.md) | Can randomized current actions control task-relevant consequences? | 320 actual randomized interventions; action-conditioned E/I8 and mediated task ranking fail all seven fixed gates under environment holdout. | Stop this pre-lift E/I8-to-Y16/32 PCA/MLP expansion after independent review; preserve weak object-rotation response and do not refute core  | probes/UNPROMISING |

## HF-actual-flow-y-ranking

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261006-actual-flow-y-ranking](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-actual-flow-y-ranking.md) | Ref14_3: actual hand flow to same-state Y ranking | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-amplitude-authority

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-amplitude-authority](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-amplitude-authority.md) | Is there a task-related interaction threshold within larger feasible actions? | Task-aligned authority UNPROMISING; localized thumb interaction response PROMISING (Probe only). | Preserve local thumb control evidence; no registered task-aligned threshold, so conditional Stage2 and neural fitting are not activated. | probes/COMPLETED |

## HF-conditional-consequence

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-conditional-consequence](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-conditional-consequence.md) | Can predicted action consequences retain oracle task information? | UNCLEAR: action-sensitive I predictions, but A/B/C fail; predicted consequences retain31.18% oracle gain without stable added value over dir | Preserve action sensitivity; stop this fixed fit without selector/PPO, diagnose generalization rather than append epochs or rescue with shuf | probes/UNCLEAR |

## HF-consequence-act-proposal

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-act-deployment-conditioned](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-deployment-conditioned.md) | Does deployment-conditioned history improve the native 24-step proposal? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261009-act-native-chunk](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-native-chunk.md) | Does a one-shot native 24-step proposal have a usable behavior contract? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261009-act-receding1](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-receding1.md) | Can one newest ACT action per query recover grasp? | UNPROMISING: newest horizon0 control every step still held0/lift0; teacher held484 with exact native dispatch. | Keep receding1 as a diagnostic mode; prioritize deployment-history diagnosis, with no retraining or further inference screens in this Probe. | probes/UNPROMISING |
| [P-20261009-act-single-open-loop-behavior](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-single-open-loop-behavior.md) | Can a single native GPU environment execute one frozen ACT chunk? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261009-act-temporal-ensemble](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-act-temporal-ensemble.md) | Does inference-only temporal aggregation recover ACT grasp behavior? | UNPROMISING aggregation-only remedy: held478 open_loop24; held0 receding8/overlap8/temporal1, with passing native execution checks. | Keep tested inference modes; prioritize deployment-history diagnosis without further aggregation-only launches or retraining in this Probe. | probes/UNPROMISING |

## HF-consequence-baseline-rebuild

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-consequence-baseline-rebuild](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261007-consequence-baseline-rebuild.md) | Rebuild a self-trained rollout substrate after asset loss | Three roles pass the operational gate (airplane_base36/64, duck8/64, cup63/64); mixed12/train5/balanced5 remain weak at 0/5/4. A six-hash ob | Freeze the six endpoint hashes and preserve weak-role runs as observational candidates; do not append unbounded specialist epochs. Keep eval | probes/PROMISING |

## HF-consequence-gate1-gt-progress

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-gate1-gt-progress](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-gate1-gt-progress.md) | Does causal physical-reference GT value improve real rolling control? | One matched pair completed in314.94s; all exact replay checks passed. | UNCLEAR. Stop CPU panel expansion and evaluator fit; restore | probes/UNCLEAR |

## HF-consequence-official-generator

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-official-generator-screen](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-official-generator-screen.md) | Can the archived DExplore actor supply useful airplane rollouts? | Both64-episode arms completed. Official lift5/hold45 coverage is64/64 | Official grasp coverage is promising, but the original perpetual-hold | probes/UNCLEAR |

## HF-consequence-oracle-headroom

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-consequence-oracle-headroom](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261007-consequence-oracle-headroom.md) | Does physical future add ranking information to decision-known residual plans? | Six-route continuous collection completed, but strict state-matched labels are insufficient for fitting: train/val/test contain 0/0/2 local  | Repeated waves make the label gate READY at 2/1/5 train/val/test pairs, but this is too sparse for a useful fit; stop before evaluator train | probes/UNCLEAR |
| [P-20261008-consequence-pair-coverage](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-consequence-pair-coverage.md) | Can repeated waves recover strict current-state preference coverage? | The historical three-wave label report was `READY` under the then-current | Stop before fitting and keep the route gate closed. The user has | probes/UNCLEAR |

## HF-consequence-physical-reference

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-physical-reference-bank](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-physical-reference-bank.md) | Does measured successful robot reference geometry fix local phase delta? | All four clean tick176 windows become positive; clean negative windows | Retain physical reference bank for a same-state GT-value Gate1 Probe; | probes/PROMISING |

## HF-consequence-reference-progress

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-reference-progress-labels](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-reference-progress-labels.md) | Can original-reference temporal progress label executed candidates causally? | Causal geometry labels pass self/static/prefix checks, but corrected | Preserve both label runs; test learned TCC alignment before Gate1. | probes/UNCLEAR |

## HF-consequence-tcc-phase

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-tcc-phase-alignment](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-tcc-phase-alignment.md) | Can a learned TCC geometry embedding recover useful reference progress? | Learned alignment recovers all four clean final phases, but local | Preserve the fixed checkpoint and labels; inspect repeated-lift and | probes/UNCLEAR |

## HF-consequence-value-data

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-full-reference-value-data](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-full-reference-value-data.md) | Full-reference outcome supervision with controlled normal placing | Completed64nominal episodes; geometry weak labels give62success/2failure, | The user explicitly chose complete reference and normal placing as | probes/PROMISING |

## HF-consequence-value-dose

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-value-dose-calibration](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-dose-calibration.md) | Can larger native-bounded residual doses support outcome learning? | Completed192episodes, class counts41/23train,38/26val,36/28test, | Separately calibrate requested dose after0.2error replay failed | probes/PROMISING |

## HF-consequence-value-interventions

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-value-contact-interventions](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-contact-interventions.md) | Is initial contact more informative than already stable holding? | Train coverage passed (59success/5failure), but val64/0and test63/1 | Both placing and stable-held empirical banks reached physics but | probes/UNCLEAR |
| [P-20261008-value-place-interventions](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-place-interventions.md) | Can empirical placing deviations supply informative outcome supervision? | Completed64episodes; clean32/32success, placing31/32success. All32plans | The user authorized continuing targeted data collection. Keep the | probes/UNPROMISING |
| [P-20261008-value-policy-error-interventions](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261008-value-policy-error-interventions.md) | Does same-H learner/expert control error supply outcome negatives? | Same-H replay passed;64episodes completed, clean31/32success and hold | Continue the authorized data-generation task with a measured policy | probes/UNPROMISING |

## HF-contact-innovation

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-contact-innovation](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-contact-innovation.md) | Does bounded surface-relative motion add information beyond joint/finger flow? | UNPROMISING for the fixed bounded basis / nuisance / ridge contract. | Preserve the completed fixed-fit result; pause the old route while ref11 proceeds on the original branch. | probes/UNPROMISING |

## HF-contactpose-transport-auxiliary

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-contactpose-transport-auxiliary](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-contactpose-transport-auxiliary.md) | Does a separate rigid-transport auxiliary help measured dynamic prediction? | Main-three completed and two-update engineering r2 passes; the300-update Probe fails when a DataLoader worker aborts after control update112 | Preserve the failed run; repair the data-loader execution before any bounded retry, keeping the frozen scientific comparison unchanged. | probes/UNCLEAR |

## HF-egodex-shared-track-admission

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-egodex-shared-real-processing](../../src/task/video-data-preparation/docs/experiments/probes/P-20261013-egodex-shared-real-processing.md) | 已下载 EgoDex 的 Shared-WAM 真实处理小探针 | UNCLEAR overall. 三个真实窗口完成，1/3 通过数值门；白色积木 32/32 endpoint 有效，位移中位数170mm；另两段在观察尺度标定处拒绝。只有 TRAIN=1、dev=0，不扩训。 | 保留真实数据处理与 Shared-WAM 接入；先修/区分尺度标定及时间 mask 覆盖问题，不直接扩大下载集处理或开始训练。 | probes/UNCLEAR |
| [P-20261014-egodex-shared-batch-ingestion](../../src/task/video-data-preparation/docs/experiments/probes/P-20261014-egodex-shared-batch-ingestion.md) | 远端 Shared-WAM 批量接入 | UNCLEAR for automatic semantic admission. 首批50段/441窗完成，74数值候选；目标误选仍明显。r4抽查16窗批准5、拒绝10、延期1；两个VAL候选均误选。批量工程链路与Shared-WAM读取已跑通。 | 保持现有几何门，用完整对象修饰词、透明物体延期和修订颜色门处理第二个不重叠50段cohort；全部自动候选仍须审查，不训练或全量扩量。 | probes/UNCLEAR |

## HF-egotouch-pressure-contact

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-egotouch-pressure-auxiliary](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-egotouch-pressure-auxiliary.md) | EgoTouch processed pressure/contact auxiliary | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-egotouch-processed-action](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-egotouch-processed-action.md) | Action-conditioned processed pressure/contact probe | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-egotouch-processed-history](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-egotouch-processed-history.md) | Causal processed-pressure history probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-egotouch-visual-context

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-egotouch-visual-context](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-egotouch-visual-context.md) | Paired visual context after the shape-only teacher screen | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-epic-contact-overlap

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-epic-contact-overlap](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-epic-contact-overlap.md) | Same-clip EPIC-Contact and ObjectForesight bridge | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-epic-scene-flow

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-epic-scene-flow-p03-03](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-epic-scene-flow-p03-03.md) | EPIC scene-flow conversion without object-pose supervision | 见原卡 | 见原卡 | probes/PROMISING |

## HF-execution-geometry

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-execution-geometry](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-execution-geometry.md) | Is nominal execution the missing geometric action contract? | UNPROMISING for the corrected fixed endpoint spatial contract. Original forecast-derived negatives are invalid. | Stop repeated fitting of this total-endpoint spatial contract; retain predictable finger execution and separate common wrist motion in the n | probes/UNPROMISING |

## HF-frozen-critic-ranking

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261006-frozen-critic-ranking](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-frozen-critic-ranking.md) | Can frozen PPO critic replace short-Y candidate ranking? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-generated-tau-execution

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-generated-tau-interface-diagnosis](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-interface-diagnosis.md) | Does the online tau execution interface preserve the GT acquisition control? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261010-generated-tau-native-execution](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-native-execution.md) | Can pure measured-history generated tau drive the frozen native executor? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-geometric-innovation

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-geometric-innovation](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-innovation.md) | Ref10: does intended surface motion generalize beyond action categories? | UNPROMISING: fixed PCA/bilinear ridge nominal-flow screen extrapolates badly; independent review plus frozen replay find no fatal wiring bug | Stop the unconstrained PCA/bilinear endpoint route, retain physical action contracts and local spatial/execution hypotheses; do not enter se | probes/UNPROMISING |
| [P-20261005-geometric-support](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-geometric-support.md) | Separate physical point-flow from feature amplification | UNPROMISING: additive physical-scaled flow avoids the bilinear explosion but still underperforms State, Arm and Joint; the isolated original | Stop PCA/ridge endpoint fitting; retain geometry contract and require local spatial or execution inductive bias for the next geometry experi | probes/UNPROMISING |

## HF-goal-point-wam-curation

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-goal-point-wam-actor-curation-ab](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-actor-curation-ab.md) | Matched decoder-only actor weighting Probe | UNPROMISING for this fixed decoder-only recipe: final B improves the | retain the original model and both Probe artifacts; stop weighting | probes/UNPROMISING |
| [P-20261013-goal-point-wam-ref3-curation-audit](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref3-curation-audit.md) | ref3: is actor-specific curation justified on the local native three domains? | UNCLEAR for a causal benefit of cleaning. Target-specific actor weighting | prioritize OakInk2 target/hand association and native pre-contact | probes/UNCLEAR |

## HF-goal-point-wam-egodex-auxiliary

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261012-goal-point-wam-egodex-auxiliary](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-egodex-auxiliary.md) | Native EgoDex hands as auxiliary training data | UNCLEAR overall; local native admission, four-step smoke and200-step | local positive learning signal supports remote same-recipe admission | probes/UNCLEAR |

## HF-goal-point-wam-object-motion-goal

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-goal-point-wam-video-object-goal](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-video-object-goal.md) | Video object-motion goal admission | 22 windows audited;12 current polygons;3 TRAIN/0 held numerical passes, all near-static supports. | Do not train; specify manipulated parts and acquire denser target tracks before matched actor probes. | probes/UNPROMISING |

## HF-goal-point-wam-ref12-methods

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261012-goal-point-wam-ref12-methods](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref12-methods.md) | ref1_2 intermediate literature methods on remaining GPUs | PROMISING for the four additional-training candidates on the shared panel; all four reached fixed3000 and passed their predeclared source ga | Retain the candidate checkpoints, prioritize intermediate-plan coverage and recheck the world-only candidate's action reliability before L_g | probes/PROMISING |

## HF-goal-point-wam-reference-goal

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-goal-point-wam-reference-wrist-goal](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-reference-wrist-goal.md) | Sparse planned wrist conditioning matched Probe | UNPROMISING for this fixed sparse-wrist continuation recipe; exact-C | preserve current parent/R, do not promote or extend this candidate; | probes/UNPROMISING |

## HF-goal-point-wam-three-domain

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-goal-point-wam-controlled-goal](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-controlled-goal.md) | Controlled multi-goal engineering diagnosis | UNCLEAR under the full engineering gate; input scaling strongly improves conditional response but not absolute teacher accuracy. | Defer real-data scale-up; inspect an explicit paired displacement encoding in a separate matched engineering Probe. | probes/UNCLEAR |
| [P-20261011-goal-point-wam-learnability](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-learnability.md) | Fixed-train-window WAM learnability diagnosis | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261011-goal-point-wam-object-goal-response](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-object-goal-response.md) | Object start/goal point contract and frozen response Probe | UNCLEAR: all three sources fit, but symmetric object-goal response remains weak. | Stop scale-up; preserve the object-goal implementation and diagnose geometric conditioning before further training. | probes/UNCLEAR |
| [P-20261011-goal-point-wam-optimizer-scale](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-optimizer-scale.md) | Matched optimizer-scale Decision Probe | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261011-goal-point-wam-ref1-alignment](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-ref1-alignment.md) | Ref1 alignment: pretrained scene and full executed-action memory | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261011-goal-point-wam-three-domain-probe](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-three-domain-probe.md) | Goal-Point-WAM three-domain no-video Probe | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261011-goal-point-wam-three-domain-smoke](../../src/task/goal-point-wam/docs/experiments/probes/P-20261011-goal-point-wam-three-domain-smoke.md) | Goal-Point-WAM three-domain engineering smoke | 见原卡 | 见原卡 | probes/COMPLETED |
| [P-20261012-goal-point-wam-compact-forecast](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-compact-forecast.md) | Compact real-input forecasting diagnostic | PROMISING for compact observed-input prediction; object-goal incremental utility remains UNCLEAR. | Retain the compact forecasting diagnostic; next use matched WAM hand-query routing to localize the representation gap, with world-loss attri | probes/PROMISING |
| [P-20261012-goal-point-wam-ddp-scale](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md) | Single-model four-GPU Goal-Point-WAM scale-up | PROMISING (local fixed final10000 Probe gate): real goal beats static goal and zero hand by>=5% on all three sources. Original remote main s | Run the selected architecture at larger global batch and fixed budget; | probes/PROMISING |
| [P-20261012-goal-point-wam-goal-injection](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-goal-injection.md) | Matched object-goal injection routes | UNCLEAR for direct-query superiority; all three larger arms and frozen audits complete. Memory goal has a local conditional-utility signal o | Stop query-only injection tuning; retain memory goal and test explicit observed-hand access. Continue deferring closed-loop world goal loss. | probes/UNCLEAR |
| [P-20261012-goal-point-wam-hand-goal-interaction](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-hand-goal-interaction.md) | Observed hand access, goal route and world auxiliary loss | PROMISING for memory observed-hand access and memory/query goal utility on the specified panel; world-loss-off remains UNCLEAR. Original raw | Retain observed-hand + memory goal + world_weight1 as the next candidate; stop query-only/world-off tuning. Validate the positive components | probes/PROMISING |
| [P-20261012-goal-point-wam-hand-query](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-hand-query.md) | Current-hand history and direct query conditioning Probe | UNCLEAR; observed velocity forecasting helps OakInk2 but fails the multi-source gate. | Stop the conditional full-WAM query training here; use a separately registered compact predictor to test input learnability. | probes/UNCLEAR |
| [P-20261012-goal-point-wam-paired-geometry](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-paired-geometry.md) | Paired goal geometry engineering comparison | UNCLEAR; neither arm passes full gate and paired displacement offers only2.3% macro analytic error reduction. | Stop analytic representation tuning; retain endpoint mode and reassess the real-data effect of coordinate units. | probes/UNCLEAR |
| [P-20261012-goal-point-wam-real-goal-scale](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-real-goal-scale.md) | Real-data goal coordinate scale comparison | UNCLEAR; matched units-only training gives no real-data trajectory improvement and negligible goal use. | Stop units-only and analytic tuning; prioritize current-hand spatial/history conditioning before longer training. | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-approach-pd](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-approach-pd.md) | Early recovery PD supervision with nominal trajectory replay | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-execution](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-execution.md) | First live WAM -> frozen R -> native PD simulated closure | 见原卡 | user ref2's hard engineering gates are named semantic mapping and | probes/UNPROMISING |
| [P-20261012-goal-point-wam-ref2-head-calibration](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-head-calibration.md) | Robot-domain trajectory head calibration with frozen WAM scene and R | 见原卡 | distinguish whether a small explicit robot-domain head adaptation can | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-joint-retargeter](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-joint-retargeter.md) | Jointly fine-tune the complete WAM policy chain and R | **UNCLEAR** for the joint policy route; mixed offline signal, no physics | retain the jointly differentiable implementation; do not close the | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-recovery-geometry](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-recovery-geometry.md) | Direct expert recovery geometry before another capacity change | direct geometry fits training and retains nominal; cross-takeover nearest8 improves23.70->19.63mm but fails predeclared halving, so r1 is no | aggregate the audited t32 true expert geometry into training explicitly, then diagnose a fresh actor's actual states under nominal retention | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-recovery-retargeter](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-recovery-retargeter.md) | Retargeter coverage on actually executed expert recovery trajectories | fixed1000 updates reduce recovery train53.22->6.05mm but held17.99->19.74mm; startup rotation retention gate fails, so no deployment. | test ONE predeclared alpha=.5 interpolation of base/final net weights with all original screens unchanged; no repeat training or rewrite of  | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-recovery](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-recovery.md) | Expert recovery from the measured WAM approach state | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-robot-aggregation](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-robot-aggregation.md) | True autonomous-state aggregation with a later expert takeover | later takeover32 locally recovers139 held frames; aggregation improves pure training636->143mm but nominal PD59.48mm fails50mm collection ga | do not deploy this head; test frozen R on actual expert-only recovery hand futures before further actor/data scaling. | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-scene-thaw](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-scene-thaw.md) | Partial PTv3 output-stage adaptation on audited recovery PD | 100 updates improve train PD44.24->33.32mm but validation384.76->398.67mm; nominal PD46.82mm misses retention; UNCLEAR, no closure. | inspect unsupported pure-WAM recovery states and supervision/replay conflict before expanding thaw depth; do not thaw the full backbone on t | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref2-shadow](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-shadow.md) | Frozen WAM -> R -> PD shadow admission | 见原卡 | test whether the final10000 human-data WAM candidate can consume | probes/UNPROMISING |
| [P-20261012-goal-point-wam-ref2-startup-anchor](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-startup-anchor.md) | Preserve startup while learning earlier true recovery geometry | startup retention restored, but the fresh autonomous actor still has0held; this fixed local recipe is UNPROMISING for immediate closed-loop  | preserve original startup and both actual-state collections; stop further geometry-only capacity/weight sweeps and inspect early visited-sta | probes/UNPROMISING |
| [P-20261012-goal-point-wam-ref2-startup](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-startup.md) | Startup coverage before live WAM-R continuation | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261012-goal-point-wam-ref2-visited-pd](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref2-visited-pd.md) | Conditional live-state PD correction through the frozen retargeter | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref22-gt-anchored](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-gt-anchored.md) | ref2_2: GT-anchored R and control-aware WAM | **PROMISING** for preserving the local GT-tau execution mapping; | 见原卡 | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref22-multitakeover](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-multitakeover.md) | ref2_2: actual student-state multi-takeover aggregation | 见原卡 | actual t8/t16 and partial t32 recovery provide admissible broader | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref22-r-horizon](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref22-r-horizon.md) | ref2_2: GT-trajectory execution-horizon diagnostic | **UNCLEAR** on the short-execution-interface hypothesis. Five matched | 见原卡 | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref23-multitakeover-fit](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref23-multitakeover-fit.md) | ref2_3: anchored multi-takeover continuation | 见原卡 | retain this candidate and evidence, preserve the Probe30 baseline, | probes/UNCLEAR |
| [P-20261012-goal-point-wam-ref24-chunk-R](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref24-chunk-R.md) | ref2_4 A: GT trajectory to independent native action chunk | Corrected128-control comparison: teacher73held, qualified oldR75held, new Chunk-R0held; zero clipping, independent label/execution audit PAS | Stop B/C under ref2_4's GT execution gate; preserve the fixed Chunk-R recipe and diagnose first-action accuracy before any WAM change. | probes/UNPROMISING |
| [P-20261012-goal-point-wam-ref25-control-aware-R](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-control-aware-R.md) | ref2_5 A2: first-action and PD-aware Chunk-R | Matched3000 updates reduce report first-PD XYZ p95 31.09->19.51mm; GT teacher/oldR/newR held73/75/74, zero clipping; GT gate PASS. | Preserve the qualified control-aware Chunk-R and proceed to the already authorized explicit-hand-readout offline comparison; C remains gated | probes/PROMISING |
| [P-20261012-goal-point-wam-ref25-hand-readout](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref25-hand-readout.md) | ref2_4 B after ref2_5: explicit observed-hand PTv3 readout | Matched200 updates per arm: nominal h1 6.51->8.59mm, first8 16.06->25.00mm; startup control improves, but10/21 non-regression checks pass. | Stop C for this fixed B recipe; retain qualified Chunk-R and both B packets, without ruling out better hand-readout adaptation. | probes/UNPROMISING |
| [P-20261012-goal-point-wam-ref26-gated-hand](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-ref26-gated-hand.md) | ref2_6 B2: zero-init gated hand residual | Initial outputs exactly preserved; matched200 updates pass21/21 checks. Nominal/startup first-PD p95 improve11.30%/18.04%; direct hand-featu | Retain B2 as a safe adapter candidate and both packets; preserve qualified A2 Chunk-R. C stays paused; do not attribute most paired gains to | probes/PROMISING |
| [P-20261012-goal-point-wam-video-points](../../src/task/goal-point-wam/docs/experiments/probes/P-20261012-goal-point-wam-video-points.md) | Shared backbone and direct video point supervision | UNCLEAR; engineering wiring passed; scientific label admission pending. | retain point-supervision implementation if wiring passes; admit a | probes/UNCLEAR |
| [P-20261013-goal-point-wam-progress-reference](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-progress-reference.md) | Object-progress goals on the admitted realign actor | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261013-goal-point-wam-ref27-effect-consistency](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref27-effect-consistency.md) | ref2_7 C: frozen-world predicted-trajectory consistency | COMPLETED at50500 total updates (C0 500 + C1 50000), global400/4GPUs. PROMISING for offline mixed-domain adaptation; Cm causal/physical util | Preserve the completed candidate and frozen A/B baselines. No further overnight training; retain the forecast tradeoffs and amended admissio | probes/PROMISING |
| [P-20261013-goal-point-wam-ref27-rollout](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref27-rollout.md) | C final native rollout: can the trained actor lift and hold? | COMPLETED542 controls; final C actor never lifts/held0, while teacher/GT-Chunk hold434/441 frames. Current checkpoint rollout is UNPROMISING | Preserve the valid negative rollout; inspect contact-region closed-loop drift before further scaling. This does not refute C or establish Cm | probes/UNPROMISING |
| [P-20261013-goal-point-wam-ref28-control-context](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref28-control-context.md) | Robot control context and current C-state recovery | Adapter2000 + decoder5000 updates completed; selected stage2 step4500. | Preserve the adapter/candidate and successful engineering contracts; | probes/UNPROMISING |
| [P-20261013-goal-point-wam-ref31-actor-weight](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref31-actor-weight.md) | Independent OakInk actor-weight short A/B | Completed paired300 updates; target-interaction actor13.189->13.167mm | Preserve implementation and evidence; no larger weighting run now. | probes/UNCLEAR |
| [P-20261013-goal-point-wam-ref31-own-R](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-ref31-own-R.md) | Chunk-R repair on actual C states | Fitted GT-to-command contract PROMISING; direct frozen-actor R swap | Preserve repaired R as an inverse-control candidate, retain original | probes/UNCLEAR |
| [P-20261013-goal-point-wam-student-pd](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-student-pd.md) | One student-state action round on the successful reference contract | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261013-goal-point-wam-visited-joint-anchor](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-visited-joint-anchor.md) | Joint actor/R update with a GT-tau action anchor | COMPLETED 3000 updates in 609s on GPU2. No checkpoint passed the | Stop this clocked joint update. Do not deploy the pair. The next | probes/UNPROMISING |
| [P-20261013-goal-point-wam-visited-realign](../../src/task/goal-point-wam/docs/experiments/probes/P-20261013-goal-point-wam-visited-realign.md) | Current visited-state coverage and fixed-R actor realignment | 27 actual takeovers collected;19 successful recovery queries admitted;R repaired/GT-qualified;actor realignment running. | Complete one collect/qualify/align/rollout loop, without larger models. | probes/RUNNING |

## HF-gt-consequence-sufficiency

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-gt-consequence-sufficiency](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md) | Do GT physical consequences preserve task-relevant action information? | GT prognosis PROMISING (46.25% primary error gain); held-arm EI bridge positive; full sufficiency UNCLEAR because remaining-action uncertain | Preserve task-relevant E/I evidence and qualify the next conditional-predictability question; no claim of identified sufficiency, online sel | probes/UNCLEAR |

## HF-gt-hand-retargeter

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-gt-hand-retargeter](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gt-hand-retargeter.md) | GT hand geometry to learned native control | UNCLEAR: the execution gate failed; recorded GT-target servo tracks within4.35mm but held117 versus teacher484, while the frozen learned ret | Keep ACT/G paused and do not train H-to-hand or connect PW/evaluator; diagnose contact execution and the learned inverse separately before r | probes/UNCLEAR |

## HF-gt-interaction-aux

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261006-gt-interaction-aux](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-gt-interaction-aux.md) | GT interaction supervision of PPO actor representation | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-hand-action-retarget

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-finger-command-coverage](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-finger-command-coverage.md) | Do the structured rollouts identify commanded finger/contact labels? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-hand-action-contact-context](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-contact-context.md) | Can current-state contact proxies improve full-action retargeting? | `UNPROMISING`. A matched v2 contextual baseline and v3 contact arm | 见原卡 | probes/UNPROMISING |
| [P-20261010-hand-action-fixed-wrist-finger](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-fixed-wrist-finger.md) | Does a fixed wrist decoder expose a viable learned finger command branch? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-hand-action-fixed-wrist-teacher-finger](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-fixed-wrist-teacher-finger.md) | Can live teacher fingers rescue the fixed-wrist native execution? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-hand-action-retarget-data](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-retarget-data.md) | Structured full-action rollouts and hand retargeter | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-hand-action-teacher-action-reference](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-hand-action-teacher-action-reference.md) | Can same-process test envs replay the complete live teacher action? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-paired-triplet-contact-hold](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-paired-triplet-contact-hold.md) | Can triplet prefixes provide matched contact and hold command rows? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-serial-finger-pulse-prefix](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-serial-finger-pulse-prefix.md) | Can a deterministic serial prefix capture a commanded-finger contact perturbation? | 见原卡 | 见原卡 | probes/PROMISING |

## HF-hand-execution-bridge

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-hand-execution-bridge](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hand-execution-bridge.md) | Can ordinary e260 interventions bridge plans to useful future hand geometry? | Prospective G->PW->C1 rolling chain completed; policy utility UNCLEAR. | Stop expanding fork panels and observed-hand-only C2a. Freeze old | probes/UNCLEAR |

## HF-history-tau-proposal

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-generated-tau-score-feasibility](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-generated-tau-score-feasibility.md) | Does generated tau preserve score information and native hand geometry? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-history-tau-proposal-diagnosis](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-tau-proposal-diagnosis.md) | Diagnose the history-to-candidate-tau blocker | 见原卡 | retain the **future-reference-free measured-H displacement baseline** | probes/PROMISING |

## HF-hocap-frame-generalization

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261008-hocap-frame-generalization](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-hocap-frame-generalization.md) | Can the fixed main-three model predict HOCap beyond a static baseline? | Fixed mixed endpoint improves moving h24 EPE22.421→17.805mm versus Oak-only (20.59%), but near-static worsens8.039→22.160mm. Same192windows; | Preserve moving-window endpoint benefit and near-static regression separately; no causal data-mixture claim due to extra training/loss chang | probes/UNCLEAR |

## HF-hold-duration-response

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-early-hold-duration](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-early-hold-duration.md) | Does longer feedback-residual execution change grasp-retention information? | 1006 full randomized windows; dose and support pass, but no duration-sensitive retention-I gate; hand displacement responds while I16/contac | Close K4/8/16 feedback-residual extension after independent review; no Cm fitting, extra duration/seed or selector. | probes/UNPROMISING |

## HF-native-gpu-execution-contract

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-gpu-cuda-sync](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-cuda-sync.md) | CUDA synchronisation as a native GPU replay probe | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261009-hidden-physx-binary-api-audit](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-binary-api-audit.md) | Does the archived native Isaac Gym binary expose a hidden-state fork/restore bridge? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261009-hidden-physx-state-api-audit](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-hidden-physx-state-api-audit.md) | Does the public Isaac Gym API expose a hidden PhysX fork/restore contract? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261009-ref13-panel-inventory](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-ref13-panel-inventory.md) | Is the complete ref13 candidate panel recoverable? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-oakink2-action-effect-wm30

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-oakink2-wm30-k24](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-oakink2-wm30-k24.md) | OakInk2 observational action-conditioned multi-object WM | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-object-relative-gt-servo

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-object-relative-gt-servo](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-object-relative-gt-servo.md) | Object-relative GT geometry and state-conditioned PD control | UNCLEAR: wrist geometry with calibrated dynamics is PROMISING; full11-point cold control does not confirm, and native finger/contact decodin | Freeze the promising analytic wrist decoder; next isolate commanded finger/contact targets. Keep old R, ACT, G and H-to-V training paused pe | probes/UNCLEAR |

## HF-official-reference-coverage

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-official-reference-screen](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-official-reference-screen.md) | Current corrected12 references: official actor physical screening | 见原卡 | candidate competent sources s3/s9 airplane, toothpaste and waterbottle; | probes/UNCLEAR |

## HF-official-video-data

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-official-video-data-subset](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-official-video-data-subset.md) | Can a bounded official DROID video subset enter the native data route? | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-old-utility-evaluator

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-old-utility-evaluator](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-old-utility-evaluator.md) | Can GT and frozen PointWorld geometry recover the old teacher's action ranking? | GT ranking screen PROMISING; C1.75641 versus C0.60256, C2.69231. | user ref4_4 freezes old U32 and authorizes matched C0/C1/C2 evaluator | probes/PROMISING |

## HF-open-video-backbone-transfer

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-open-video-backbone-transfer](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-open-video-backbone-transfer.md) | Historical checkpoint-transfer run (not the requested official-video-data test) | 见原卡 | do not use this run to stop the video-data route. Preserve its outputs, | probes/UNCLEAR |

## HF-openaoe-shared-track-admission

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261014-openaoe-local-admission](../../src/task/video-data-preparation/docs/experiments/probes/P-20261014-openaoe-local-admission.md) | Open-AoE 本地子集接入探针 | PROMISING（仅手/相机接入）。两段真实30Hz视频完成MANO解码、H4/K24合同检查，用户确认抽样投影基本贴合；完整手物训练准入仍UNCLEAR。 | 保留有界reader/audit，下一步用刚性容器/瓶子做少量完整手物窗口；先适配当前目标语义与坐标，暂不扩大下载量。 | probes/PROMISING |

## HF-oracle-flow-task

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-oracle-flow-task](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-flow-task.md) | Ref12: oracle hand flow → OOF predicted E/I → task Y | UNPROMISING for the registered fixed-fit chain; direct oracle flow task information is PROMISING, predicted E/I increment remains uncertain. | Preserve direct flow task information; stop this fixed chain fit without planner, paired data, execution or epoch/seed extensions. | probes/UNPROMISING |

## HF-oracle-hand-flow

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-oracle-hand-flow](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-hand-flow.md) | Ref11: oracle endpoint and temporal hand flow → E/I | UNPROMISING for the predeclared joint E/I gate at this fixed budget; Chunk E passes, I remains uncertain. | Retain the oracle flow E signal and I uncertainty; paired work is deferred and ref12 tests task-information transfer on the original branch. | probes/UNPROMISING |

## HF-oracle-y-utility

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-oracle-y-utility](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-oracle-y-utility.md) | Ref13 Oracle-Y Utility Gate | UNPROMISING for the fixed selector: baseline23/32 vs GT-Y24/32, +3.125pp [0,9.375]; GT-Z candidate upper25/32. | Stop this short-Y/seven-candidate selector; GatesB/C/D not activated, no predictor or PPO fitting. | probes/UNPROMISING |

## HF-per-finger-control

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-per-finger-control](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-per-finger-control.md) | Does separating the finger synergy expose task-related interaction control? | 854 full windows; registered linked/useful gates UNPROMISING, with late-contact action response and yaw-minus/middle-minus clues retained. | Preserve physical magnitude tables and targeted task clues; no amplitude/seed sweep or Cm fitting from this completed contract. | probes/COMPLETED |

## HF-pointflow-G

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-pointflow-g](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-pointflow-g.md) | Does real point-flow predicted E preserve E-to-G information? | UNCLEAR for the valid K1 point-flow-to-G bridge; UNPROMISING for expanding a pose-only K4 predictor into this G regression from the fixed GT | keep E as the physical main route, freeze I head/K4 expansion and this frozen K1 G teacher; audit task-relevant hold/drop value evidence bef | probes/UNCLEAR |

## HF-pointwam-data-pipeline

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-pointwam-data-expansion](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-pointwam-data-expansion.md) | PointWAM official-TRAIN data expansion | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261011-pointwam-data-pipeline](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-pointwam-data-pipeline.md) | PointWAM human scene/hand data pipeline: small engineering pilot | VGGT-Omega/CoTracker3 completed two clips in97.76s; numerical gates and stored-pack audit pass; user accepts the visual preview. | proceed to bounded official-TRAIN qualification with episode-isolated fit/dev; retain the approximate palm convention. | probes/UNCLEAR |
| [P-20261011-pointwam-train-qualification](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-pointwam-train-qualification.md) | PointWAM official-TRAIN data qualification | 12/12 TRAIN clips from3tasks/6records qualify; record-isolated fit/dev each contain6clips; processing605.94s and peak GPU allocation6923.70M | retain this bounded subset for joint scene/hand learning; do not expand raw data or admit TEST samples. | probes/PROMISING |

## HF-pointwam-joint-learning

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-pointwam-joint-learning](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-pointwam-joint-learning.md) | Bounded scene/hand joint learning | UNCLEAR. The fixed main run completes and hand fitting improves, but | retain the qualified data and verified joint learning contract; | probes/UNCLEAR |
| [P-20261011-pointwam-optimization-diagnosis](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-pointwam-optimization-diagnosis.md) | Optimization diagnosis for the bounded PointWAM learner | `UNPROMISING` for the optimizer-only variant; fit scene reduction is | retain the verified data contract and stop this small substitute | probes/UNPROMISING |

## HF-pointwam-repro-human-forecaster

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-pointwam-repro-frozen-forward](../../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-frozen-forward.md) | Frozen encoders, one small forward | UNCLEAR. Frozen Mosaic3D and Recap-CLIP produced finite Volt-H displacements on 1024 points of one qualified clip. The forecaster weights ar | Do not start 60k or global batch 192. The encoder path can be reused for a later training step. | probes/UNCLEAR |
| [P-20261013-pointwam-repro-frozen-step](../../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-frozen-step.md) | Frozen encoders, one optimizer step | UNCLEAR. One AdamW step on the full qualified clip finished with finite loss 7.903 and gradient norm 5.592. The forecaster is still randomly | Do not start 60k or global batch 192. One step fits; that is not a reason to treat this clip as pre-training. | probes/UNCLEAR |
| [P-20261013-pointwam-repro-patch-fit](../../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-patch-fit.md) | Reduced patch forecaster on qualified human clips | UNCLEAR. Zero-init fit hand EPE 54.8 to 29.5 mm, but fit scene 9.0 to 15.1 mm and dev hand 69.3 to 87.1 mm. | Stop scaling this substitute. Do not add epochs or Mosaic3D until scene and held-out hand can improve together. | probes/UNCLEAR |
| [P-20261013-pointwam-repro-volt-forward](../../src/task/pointwam-repro/docs/experiments/probes/P-20261013-pointwam-repro-volt-forward.md) | Paper-sized Volt-H, one synthetic step | UNCLEAR. One bf16 step on 12000 random points finished in 1.224 s at 12776 MiB, but the features are not Mosaic3D or Recap-CLIP. | Do not start 60k or global batch 192. Load the frozen encoders before any clip training. | probes/UNCLEAR |

## HF-pointworld-multisource

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-pointworld-multisource](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-multisource.md) | Source-aware physical prediction with an OakInk2 warm start | Four-source14250 preserved; main-three COMPLETED50000, fixed-panel macro moving-anchor h24 EPE34.480→27.126mm; latest/final50000 and best460 | Preserve the completed main checkpoint; run the separate bounded history-only ContactPose auxiliary ablation, keeping EPIC non-training. | probes/UNCLEAR |

## HF-pointworld-unified-action-effect

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-pointworld-action-ddp](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-action-ddp.md) | User-directed action-only three-GPU warm start | 见原卡 | retain the improved action checkpoint. The user-directed three-rank | probes/PROMISING |
| [P-20261007-pointworld-ref4-input-loss-audit](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-ref4-input-loss-audit.md) | Ref4 action-voxel and motion-weight diagnosis | substantial cross-time action merging; low raw selector weights do not imply uniform supervision suppression after normalization. | diagnose input-time identity and relative motion-weight exposure while preserving the live three-arm run. | probes/UNCLEAR |
| [P-20261007-pointworld-small-wm24](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-small-wm24.md) | Unified spatial hand/action-to-rigid-effect Probe | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261007-pointworld-temporal-wm24](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-pointworld-temporal-wm24.md) | Time-preserving PointWorld action/effect Probe | at matched6000-update validation, H+A moving-anchor h24 point EPE is15.63mm versus H21.81mm and shuffled-action21.90mm; training and final T | evaluate observed-action predictive benefit after the user-requested ref4 correction. | probes/UNCLEAR |

## HF-ptv3-shared-rl660-joint

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261014-ptv3-shared-rl660-joint](../../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261014-ptv3-shared-rl660-joint.md) | 全660条DExplore轨迹：同执行补录与共享预测器/Chunk-R联合微调 | RUNNING. 660条/50类补录审核完成；原小模型在完整1000步后保留并停止。容量扩展到101636370参数、global512，初始预测保持与短更新稳定性门通过；大模型新phase19000步已在GPU5/7启动。 | 用户明确选择补录完整机器人执行，再联合微调WAM+Chunk-R；不只做airplane。 | probes/RUNNING |

## HF-ptv3-shared-task-forecaster

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-ptv3-shared-main3-80k](../../src/task/ptv3-shared-wam/docs/experiments/probes/P-20261013-ptv3-shared-main3-80k.md) | 四卡三域共享预测器预训练：80,000 updates | PROMISING（局部离线预测；用户已提前停止预训练）。37k固定dev显示局部学习信号：GRAB/ARCTIC目标物体中段误差低于goal-only插值；手轨迹相对保持当前手不动改善约35–39%，但GRAB后16步平均误差仍122mm。原训练最后日志39100，完整恢复点3 | 用户要求停止三域80k训练，保留完整best available/latest，转向660条机器人执行补录与联合适配。 | probes/PROMISING |

## HF-ptv3-shared-v2-pretrain

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261014-ptv3-shared-v2-pretrain](../../src/task/ptv3-shared-wam-v2/docs/experiments/probes/P-20261014-ptv3-shared-v2-pretrain.md) | 联合轻量PTv3与大shared的horizon标准化预训练 | RUNNING（远端80k阶段）。本地1000步局部PROMISING，远端双卡与恢复预检通过；按用户明确要求启动GPU0/2长训。 | 用户已明确要求直接开始训练；采用预检通过的模型/配方，在独立远端run从头训练80000更新，保留本地Probe和工程checkpoint。 | probes/RUNNING |

## HF-rawsensor-hand-conditioning

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-rawsensor-hand-conditioning](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-rawsensor-hand-conditioning.md) | Raw sensor supervision and human hand conditioning readiness | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-ref13-progress-recovery

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-ref13-progress-recovery](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261009-ref13-progress-recovery.md) | Recover ref13 whole-world cold replay before replacing Y | 见原卡 | recover the old whole-world method as a viable engineering route. | probes/UNCLEAR |

## HF-ref5-data-expansion

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261007-ref5-data-expansion](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261007-ref5-data-expansion.md) | Ref5 native-3D expansion and EgoDex engineering pilot | 见原卡 | test whether existing GRAB/ARCTIC and native EgoDex hands can feed | probes/UNCLEAR |

## HF-reference-tracking-control

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-reference-tracking-feedforward](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking-feedforward.md) | Wrist velocity feedforward for sustained reference holding | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261010-reference-tracking](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-reference-tracking.md) | Does reference-conditioned residual RL recover native object holding? | UNPROMISING for the predeclared near-teacher screen; transient holding improved (nominal median0, tracker286.5, teacher483), but all tracker | Retain the controller for a later targeted sustained-contact Probe; do not yet replace the robot-q oracle with 11-point retargeting or decla | probes/UNPROMISING |
| [P-20261010-tau-finger-canonicalization](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-finger-canonicalization.md) | Preserve the tau tracker while learning feasible finger commands | 见原卡 | learn only the six finger output rows of the existing897-input actor | probes/PROMISING |
| [P-20261010-tau-geometry-tracking](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-geometry-tracking.md) | Remove measured q and future-object reference from the execution controller | 见原卡 | repair only reward, keep geometry, original warmstart, seed277, | probes/UNPROMISING |
| [P-20261011-tau-residual-basin-repeat](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261011-tau-residual-basin-repeat.md) | One-launch repeat of the fixed tau residual basin | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261011-tau-residual-basin](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261011-tau-residual-basin.md) | Fixed residual-basin diagnostic around the retained tau executor | 见原卡 | 见原卡 | probes/PROMISING |

## HF-relative-action

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-recap-relative-action](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-recap-relative-action.md) | Can relative task outcome supervise a deployable action critic? | G0 label stability failed (train34.7%, test39.1%); counts/support and V-versus-zero checks pass; G1 not executed. | retain the relative-outcome direction but stop critic/Cm expansion on this unstable discrete label contract; no action-information or policy | probes/UNCLEAR |
| [P-20261004-relative-action-ranking](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-relative-action-ranking.md) | Does current action improve ranking of frozen relative task advantage? | H macro pair59.71%, HaK1 59.62%, HaK4 realized60.52%; current-action ranking gate fails after independent implementation review. | stop critic/Cm expansion on this frozen target/data/fit contract; proxy ordering also lacks strong V-fit stability, so do not refute physica | probes/UNPROMISING |

## HF-relative-finger-innovation

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-relative-finger-innovation](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-relative-finger-innovation.md) | Can full finger flow learn action innovations over an OOF state forecast? | UNPROMISING for the fixed anatomical summary / OOF-state innovation contract. | Stop this fixed seven-head contract; retain the physical action and Cm hypotheses. | probes/UNPROMISING |

## HF-rolling-gt-y

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-rolling-gt-y](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-gt-y.md) | Ref13_1 Rolling GT-Y audit | PROMISING recorded-path rolling observability:46/50 initially tied discordantpairs separate correctly atscheduledqueries,8/32anchors; rollin | Retain short-Y and prioritize a new same-current-state rolling oracle test; no long-Y/predictor/PPO fitting or saved-trajectory policy-gain  | probes/PROMISING |

## HF-rolling-oracle-control

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-rolling-oracle-control](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-rolling-oracle-control.md) | Ref14_1: executed Rolling GT-Y Oracle Control | PROMISING on the predeclared 32-anchor gate: baseline 23/32, actual | Execute same-current-state rolling oracle before any new predictor, long-Y, execution forecasting or PPO. | probes/PROMISING |

## HF-spatial-action-fidelity

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-spatial-action-fidelity](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-action-fidelity.md) | Where does continuous finger action information survive? | UNPROMISING for linear action recovery from the fixed local/fused paths; interpretation restricted to the existing arm dictionary. | Preserve full per-finger motion and test a centered action-innovation objective; do not infer that the 2cm radius caused consequence failure | probes/UNPROMISING |

## HF-spatial-consequence

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-spatial-consequence](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-spatial-consequence.md) | Does shared local spatial structure preserve intended action consequences? | UNPROMISING: reduced spatial encoding remains stable but preserves too little intended action contrast; OOF consequences show no unique task | Stop this fixed nominal spatial fit; retain GT prognosis and examine the nominal-to-realized execution contract before further geometry lear | probes/UNPROMISING |

## HF-tactile-data-readiness

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-egotouch-label-schema](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-egotouch-label-schema.md) | EgoTouch original label schema sample | UNCLEAR for training readiness. Acquisition completed in129.97s, with | keep tactile training gated. Next cheap tactile probe should select | probes/UNCLEAR |

## HF-trajectory-conditioned-evaluator

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-history-panel-trajectory-utility](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-panel-trajectory-utility.md) | Does an episode-split approximate-H panel let tau retain U32 ranking signal? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-history-preserving-candidate-bank](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-preserving-candidate-bank.md) | Does a history-preserving structured rollout retain H-matched candidate pairs? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261010-history-to-tau](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-to-tau.md) | Can history predict a short current-object-frame hand trajectory? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-history-utility-effect-audit](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-history-utility-effect-audit.md) | Is the frozen C1 GT-effect branch safe enough for later-chain selection? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-serial-exact-h-candidate-bank](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-serial-exact-h-candidate-bank.md) | Can serial deterministic replay create an exact-H, non-tied candidate bank? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-structured-candidate-bank-audit](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-structured-candidate-bank-audit.md) | Can the existing structured rollouts support a nontrivial candidate bank? | `UNCLEAR`. The frozen structured rollouts contain useful candidate | 见原卡 | probes/UNCLEAR |
| [P-20261010-tau-selector-abstention](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-tau-selector-abstention.md) | Can an observed-trajectory selector abstain safely near the baseline? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261010-trajectory-utility](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261010-trajectory-utility.md) | Does a real hand trajectory interface retain consequence ranking information? | `UNCLEAR`. C0 (`H+tau`) has 62.82% strict pair accuracy; C1 | 见原卡 | probes/UNCLEAR |

## HF-trajectory-policy-action-space

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-decoder-prefix-fitting](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-decoder-prefix-fitting.md) | Can fitting c preserve the immediate wrist trajectory with fixed D? | 见原卡 | do not spend another native wave on this initializer. Keep c48 and | probes/UNPROMISING |
| [P-20261010-lowrank-trajectory-decoder](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-lowrank-trajectory-decoder.md) | Can a fixed learned time basis preserve executable trajectories? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-metric-trajectory-decoder](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-metric-trajectory-decoder.md) | Does a pose/FF metric retain the executable trajectory in D48? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-trajectory-decoder-coverage](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-decoder-coverage.md) | Can a fixed48D trajectory decoder retain executable grasp behavior? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-trajectory-policy-competent-actor-cadence

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-competent-actor-cadence](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-actor-cadence.md) | Does current-state per-control trajectory replanning preserve competence? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-trajectory-policy-competent-joint-bc

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-competent-joint-bc](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-joint-bc.md) | Competent teacher H/G -> trajectory -> frozen actual-PD R | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-trajectory-policy-competent-pd

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-competent-pd-retargeter](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-pd-retargeter.md) | Competent official rollout to measured-state/trajectory-conditioned PD targets | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-trajectory-policy-competent-pd-physical

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-competent-pd-physical](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-competent-pd-physical.md) | Does the competent PD retargeter's finite error preserve real task behavior? | 见原卡 | 见原卡 | probes/PROMISING |

## HF-trajectory-policy-deployment

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-trajectory-stochastic-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-stochastic-execution.md) | Is stochastic deployment hiding a learned usable trajectory policy? | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-trajectory-policy-execution-interface

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-goal-conditioned-executor-contract](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-goal-conditioned-executor-contract.md) | Can the trajectory-policy executor expose a single closed-loop goal contract? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261011-goal-conditioned-bc](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-bc.md) | Stage 1 goal-conditioned low-level BC | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261011-goal-conditioned-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-execution.md) | Stage 1 physical execution gate | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-goal-conditioned-finger-residual](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-finger-residual.md) | Analytic-wrist, finger-only residual student | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-goal-conditioned-ppo-execution](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo-execution.md) | Frozen evaluation of the Stage 1 PPO actor | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-goal-conditioned-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-ppo.md) | Stage 1 goal-conditioned PPO continuation | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261011-goal-conditioned-residual-integration](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-integration.md) | Goal-conditioned residual policy through the shared executor | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-goal-conditioned-residual-offset](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-goal-conditioned-residual-offset.md) | Fixed thumb preload offset on the residual student | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261011-regrind-executor-gt-generated](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-regrind-executor-gt-generated.md) | Same-executor GT/generated tau physical comparison | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-trajectory-policy-history-to-tau

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-current-history-tau-forecast](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-current-history-tau-forecast.md) | Current-source pure measured-history H to tau forecast | 见原卡 | 见原卡 | probes/PLANNED |

## HF-trajectory-policy-learning

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-history-trajectory-actor](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-history-trajectory-actor.md) | Can measured history initialize a standalone trajectory actor? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-startup-balanced-actor](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-startup-balanced-actor.md) | Does startup-balanced BC repair the independent actor initializer? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-trajectory-ppo-long-credit](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-ppo-long-credit.md) | Does longer credit improve actual trajectory policy learning? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-trajectory-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-ppo.md) | Can real-task PPO improve the independent trajectory actor? | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-trajectory-policy-optimization

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-trajectory-step-replay](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-trajectory-step-replay.md) | Does persistent early LR backtracking constrain later useful updates? | 见原卡 | 见原卡 | probes/PROMISING |

## HF-trajectory-policy-pw-critic

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-pw-critic-policy](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-pw-critic-policy.md) | Frozen PointWorld action-conditioned Q and bounded policy improvement | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-trajectory-policy-reference-goal-dagger

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-reference-goal-dagger](../../src/task/trajectory-policy/docs/experiments/probes/P-20261011-reference-goal-dagger.md) | Authorized human/object reference goals and one-round PD DAgger | 见原卡 | proceed with already specified one-round PD-query correction because | probes/PROMISING |
| [VAL-20261011-reference-goal-dagger-repeat](../../src/task/trajectory-policy/docs/experiments/validations/VAL-20261011-reference-goal-dagger-repeat.md) | Frozen reference-conditioned PD DAgger: two holdout launches | 见原卡 | 见原卡 | validations/SUPPORTED |

## HF-trajectory-policy-retargeter

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-learned-retargeter-inverse](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-learned-retargeter-inverse.md) | Can actual future hand motion supervise a trajectory-dependent native retargeter? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-retargeter-local-motion](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-retargeter-local-motion.md) | Can action-aligned local hand motion improve native inverse precision? | `UNCLEAR` — local-motion improves offline τ usage, but the fixed native-precision screen still fails. | retain the local-motion evidence as `UNCLEAR`, close this fixed Probe, | probes/UNCLEAR |
| [P-20261010-retargeter-step-inverse](../../src/task/trajectory-policy/docs/experiments/probes/P-20261010-retargeter-step-inverse.md) | Does an observable local wrist inverse explain the native control precision gap? | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-unidexgrasp-rigid-admission

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261014-unidexgrasp-local-admission](../../src/task/video-data-preparation/docs/experiments/probes/P-20261014-unidexgrasp-local-admission.md) | UniDexGrasp / ++ 刚体样例接入检查 | PROMISING（单对象接入Probe）。修复物理配置后，真实GPU回放13/16达到官方目标，30Hz/H4/K24及固定32点标签审计通过；完整训练准入仍待多对象/dev与profile接入。 | 保留有界回放/审计工具，优先扩少量刚体对象、按mesh分组建立dev，明确ShadowHand/CAD profile后接训练；不将单对象成功率升级为全库结论。 | probes/PROMISING |

## HF-video-data-readiness

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-video-data-readiness](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-video-data-readiness.md) | Local video data readiness after semantic contract repairs | UNPROMISING for immediate joint hand/scene training from this local | retain OakInk2/GRAB/ARCTIC as main training sources. Do not start mixed | probes/UNPROMISING |

## HF-video-point-dynamics

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-video-point-dynamics](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-video-point-dynamics.md) | Video point prediction without dense future hand labels | 见原卡 | distinguish premature clip-start track death from intrinsically poor | probes/UNPROMISING |
| [P-20261011-video-motion-token](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-video-motion-token.md) | Metric-3D history motion-token Probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF-wam-residual-ppo

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261013-wam-residual-ppo](../../src/task/trajectory-policy/docs/experiments/probes/P-20261013-wam-residual-ppo.md) | Residual PPO on the frozen context pair | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261014-wam-residual-basin](../../src/task/trajectory-policy/docs/experiments/probes/P-20261014-wam-residual-basin.md) | Frozen finger authority screen after the corrected PPO budget/course probe | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261014-wam-residual-curriculum](../../src/task/trajectory-policy/docs/experiments/probes/P-20261014-wam-residual-curriculum.md) | More updates versus contact-focused residual curriculum | 见原卡 | 见原卡 | probes/UNCLEAR |

## HF-weak-temporal-value

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261009-weak-temporal-value](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-weak-temporal-value.md) | Can outcome/time supervision learn useful GT consequence value? | initial offline screen UNCLEAR; completed matched 1200-step fit. | user ref4_3 authorizes a bounded learned evaluator Probe, replacing | probes/UNCLEAR |

## HF-y-noise-tolerance

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261006-y-noise-tolerance](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261006-y-noise-tolerance.md) | Gate 3: offline short-Y ranking/noise tolerance | 见原卡 | 见原卡 | probes/PROMISING |

## HF02

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-temporal-expert-credit](probes/P-20260926-temporal-expert-credit.md) | Probe: temporal expert-option credit | 见原卡 | 见原卡 | probes/UNPROMISING |

## HF03

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-contact-supported-credit](probes/P-20260926-contact-supported-credit.md) | Probe: contact-supported credit from executed handflow | 见原卡 | **UNPROMISING**. Freeze HF03 after this one bounded audit. Do not | probes/UNPROMISING |

## HF04

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-trajectory-credit](probes/P-20260926-trajectory-credit.md) | Probe: short trajectory-level contact credit | 见原卡 | **UNPROMISING**. HF04 is frozen after this one CPU screen. No | probes/UNPROMISING |

## HF05

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-selective-causal-gate](probes/P-20260926-selective-causal-gate.md) | Probe: conservative selective causal Cm intervention | 见原卡 | **UNPROMISING**. Freeze HF05 after this one CPU run. Do not change the | probes/UNPROMISING |

## HF06

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260930-scratch-mlp-feasibility](probes/P-20260930-scratch-mlp-feasibility.md) | Decision prerequisite: fixed scratch MLP feasibility | `UNPROMISING` for this teacher-envelope implementation, run_status | 见原卡 | probes/UNPROMISING |

## HF07

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260930-cm-inference-bottleneck](probes/P-20260930-cm-inference-bottleneck.md) | Real-policy Probe: explicit physical prediction input | `UNPROMISING`; execution `COMPLETED` on code | close HF07; no Validation and no local width/seed/step/target sweep. | probes/UNPROMISING |

## RBO-object-wrench-adapter

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-rbo-wrench-alignment](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-rbo-wrench-alignment.md) | RBO RGB-D / FT alignment Probe | 见原卡 | 见原卡 | probes/PROMISING |

## REF5-SURFACE-I

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-ref5-surface-i-gt-value](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-ref5-surface-i-gt-value.md) | Ref5: does GT spatial surface I add G information after H and E? | 见原卡 | 见原卡 | probes/UNPROMISING |

## official-raw-video-data

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-droid100-raw-input-package-audit](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-droid100-raw-input-package-audit.md) | Does the raw DROID episode already contain the inputs needed for 3-D reprocessing? | 见原卡 | 见原卡 | probes/UNCLEAR |
| [P-20261010-droid100-raw-video-temporal-audit](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-droid100-raw-video-temporal-audit.md) | Does the matched official raw DROID episode have native temporal coverage? | 见原卡 | 见原卡 | probes/PROMISING |
| [P-20261010-droid100-video-data-audit](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-droid100-video-data-audit.md) | Can the official DROID video subset provide actual frame/action samples? | 见原卡 | 见原卡 | probes/PROMISING |

## official-video-data-only-native-transfer

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-official-video-native-transfer](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-official-video-native-transfer.md) | Official video data-only native-transfer Probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## official-video-pointflow-adapter

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261010-droid100-native-adapter-contract](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-droid100-native-adapter-contract.md) | Does the matched official H5 clip satisfy the native point-flow input contract? | 见原卡 | 见原卡 | probes/UNPROMISING |
| [P-20261010-droid100-pointworld-match](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261010-droid100-pointworld-match.md) | Can an official DROID video episode be paired with PointWorld 3-D labels? | 见原卡 | 见原卡 | probes/PROMISING |

## official-video-scene-flow-auxiliary

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-official-video-aux](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-official-video-aux.md) | Official DROID / PointWorld H5 auxiliary Probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## official-video-state-conditioned-scene-flow

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-official-video-state-aux](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-official-video-state-aux.md) | Official H5 state-conditioned auxiliary Probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## rbo-measured-wrench-temporal-auxiliary

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-rbo-wrench-auxiliary](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-rbo-wrench-auxiliary.md) | RBO measured-wrench auxiliary Probe | 见原卡 | 见原卡 | probes/UNPROMISING |

## simulator-chord-wrench-action-ranking

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261011-chord-action-ranking](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261011-chord-action-ranking.md) | Simulator CHORD action-ranking Probe | 见原卡 | 见原卡 | probes/RUNNING |

## 未分类历史记录

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260923-CM-CRITIC-SALIENCE](probes/P-20260923-CM-CRITIC-SALIENCE.md) | Probe: Cm 一步效应只引导 PPO critic 的状态显著性 | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-CM-CRITIC-SECOND-SEED](probes/P-20260923-CM-CRITIC-SECOND-SEED.md) | Probe: Cm critic 显著性在第二困难训练 seed 的复制 | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-CM-JOINT-HARD-SEED](probes/P-20260923-CM-JOINT-HARD-SEED.md) | Probe: 早期 joint Cm actor 权重能否通过失败 seed 的压力测试？ | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-CM-JOINT-SECOND-SEED](probes/P-20260923-CM-JOINT-SECOND-SEED.md) | Probe: joint Cm 权重在第二个困难训练 seed 上是否复制？ | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-CM-SIGNED-UP-RANK](probes/P-20260923-CM-SIGNED-UP-RANK.md) | Probe: Cm 有向一步效应能否改善 PPO 样本排序？ | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-mixed-cm-sim-adapt](probes/P-20260923-mixed-cm-sim-adapt.md) | P-20260923-mixed-cm-sim-adapt | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260923-mixed-cm-sim-transfer](probes/P-20260923-mixed-cm-sim-transfer.md) | P-20260923-mixed-cm-sim-transfer | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-airplane-to-apple-transfer](probes/P-20260924-airplane-to-apple-transfer.md) | P-20260924-airplane-to-apple-transfer | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-apple-failure-phase-audit](probes/P-20260924-apple-failure-phase-audit.md) | P-20260924-apple-failure-phase-audit | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-calibrated-cm-transfer](probes/P-20260924-calibrated-cm-transfer.md) | P-20260924-calibrated-cm-transfer | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-cate-ranking](probes/P-20260924-cm-cate-ranking.md) | P-20260924-cm-cate-ranking | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-crossaxis-primer](probes/P-20260924-cm-crossaxis-primer.md) | P-20260924-cm-crossaxis-primer | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-geometry-complement](probes/P-20260924-cm-geometry-complement.md) | P-20260924-cm-geometry-complement | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-h10-ppo-aux-representation](probes/P-20260924-cm-h10-ppo-aux-representation.md) | P-20260924-cm-h10-ppo-aux-representation | 见原卡 | stop this auxiliary target/coefficient. Do not tune on | probes/见原卡 |
| [P-20260924-cm-online-action-boost](probes/P-20260924-cm-online-action-boost.md) | P-20260924-cm-online-action-boost | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-online-latency](probes/P-20260924-cm-online-latency.md) | P-20260924-cm-online-latency | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-ppo-aux-representation](probes/P-20260924-cm-ppo-aux-representation.md) | P-20260924-cm-ppo-aux-representation | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-two-step-model](probes/P-20260924-cm-two-step-model.md) | P-20260924-cm-two-step-model | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-two-step-physical-effect](probes/P-20260924-cm-two-step-physical-effect.md) | P-20260924-cm-two-step-physical-effect | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-two-step-structured-model](probes/P-20260924-cm-two-step-structured-model.md) | P-20260924-cm-two-step-structured-model | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cm-two-step-z-effect](probes/P-20260924-cm-two-step-z-effect.md) | P-20260924-cm-two-step-z-effect | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cmv13-object-loo-representation](probes/P-20260924-cmv13-object-loo-representation.md) | P-20260924-cmv13-object-loo-representation | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-contact-aware-cm](probes/P-20260924-contact-aware-cm.md) | P-20260924-contact-aware-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-contact-cm-online-down](probes/P-20260924-contact-cm-online-down.md) | P-20260924-contact-cm-online-down | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-cross-object-input-gate](probes/P-20260924-cross-object-input-gate.md) | P-20260924-cross-object-input-gate | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-crossobject-cm-action-info](probes/P-20260924-crossobject-cm-action-info.md) | P-20260924-crossobject-cm-action-info | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-crossobject-contact-future-cm](probes/P-20260924-crossobject-contact-future-cm.md) | P-20260924-crossobject-contact-future-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-crossobject-finger-effect](probes/P-20260924-crossobject-finger-effect.md) | P-20260924-crossobject-finger-effect | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-duck-contact-action-opportunity](probes/P-20260924-duck-contact-action-opportunity.md) | P-20260924-duck-contact-action-opportunity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-duck-specialist](probes/P-20260924-duck-specialist.md) | P-20260924-duck-specialist | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-executed-handflow](probes/P-20260924-executed-handflow.md) | P-20260924-executed-handflow | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-expert-multiaxis-choice](probes/P-20260924-expert-multiaxis-choice.md) | P-20260924-expert-multiaxis-choice | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-expert-option-availability](probes/P-20260924-expert-option-availability.md) | P-20260924-expert-option-availability | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-expert-physical-cm-data](probes/P-20260924-expert-physical-cm-data.md) | P-20260924-expert-physical-cm-data | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-finger-primer-lift](probes/P-20260924-finger-primer-lift.md) | P-20260924-finger-primer-lift | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-intervention-handflow](probes/P-20260924-intervention-handflow.md) | P-20260924-intervention-handflow | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-local-geometric-cm](probes/P-20260924-local-geometric-cm.md) | P-20260924-local-geometric-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-multiaxis-h10-cm](probes/P-20260924-multiaxis-h10-cm.md) | P-20260924-multiaxis-h10-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-multiaxis-h10-effects](probes/P-20260924-multiaxis-h10-effects.md) | P-20260924-multiaxis-h10-effects | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-multiaxis-h10-online-choice](probes/P-20260924-multiaxis-h10-online-choice.md) | P-20260924-multiaxis-h10-online-choice | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-multitrajectory-baseline](probes/P-20260924-multitrajectory-baseline.md) | P-20260924-multitrajectory-baseline | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-object-pool-audit](probes/P-20260924-object-pool-audit.md) | P-20260924-object-pool-audit | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-object-specialist-router](probes/P-20260924-object-specialist-router.md) | P-20260924-object-specialist-router | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-paired-sim-actions](probes/P-20260924-paired-sim-actions.md) | P-20260924-paired-sim-actions | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-randomized-action-effect](probes/P-20260924-randomized-action-effect.md) | P-20260924-randomized-action-effect | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-randomized-contact-followup](probes/P-20260924-randomized-contact-followup.md) | P-20260924-randomized-contact-followup | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-randomized-geometric-cm](probes/P-20260924-randomized-geometric-cm.md) | P-20260924-randomized-geometric-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-raw-effect-cm-heldout](probes/P-20260924-raw-effect-cm-heldout.md) | P-20260924-raw-effect-cm-heldout | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-reference-multitrajectory-feasibility](probes/P-20260924-reference-multitrajectory-feasibility.md) | P-20260924-reference-multitrajectory-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-s1-task-aligned-option](probes/P-20260924-s1-task-aligned-option.md) | P-20260924-s1-task-aligned-option | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-s3-expert-outcome-cm](probes/P-20260924-s3-expert-outcome-cm.md) | P-20260924-s3-expert-outcome-cm | 见原卡 | do not run an online route or tune this model on seeds304/305. | probes/见原卡 |
| [P-20260924-selftrained-object-split](probes/P-20260924-selftrained-object-split.md) | P-20260924-selftrained-object-split | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-stratified-e320-support](probes/P-20260924-stratified-e320-support.md) | P-20260924-stratified-e320-support | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-sustained-grip-lift-option](probes/P-20260924-sustained-grip-lift-option.md) | P-20260924-sustained-grip-lift-option | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-temporal-contact-cm](probes/P-20260924-temporal-contact-cm.md) | P-20260924-temporal-contact-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-train5-balanced-coverage](probes/P-20260924-train5-balanced-coverage.md) | P-20260924-train5-balanced-coverage | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-train5-causal-cm](probes/P-20260924-train5-causal-cm.md) | P-20260924-train5-causal-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-train5-control](probes/P-20260924-train5-control.md) | P-20260924-train5-control | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-train5-state-coverage](probes/P-20260924-train5-state-coverage.md) | P-20260924-train5-state-coverage | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-train8-h10-action-value](probes/P-20260924-train8-h10-action-value.md) | P-20260924-train8-h10-action-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-waterbottle-learnability](probes/P-20260924-waterbottle-learnability.md) | P-20260924-waterbottle-learnability | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-waterbottle-specialist](probes/P-20260924-waterbottle-specialist.md) | P-20260924-waterbottle-specialist | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260924-waterbottle-start-anneal](probes/P-20260924-waterbottle-start-anneal.md) | P-20260924-waterbottle-start-anneal | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-balanced-observation-router](probes/P-20260925-balanced-observation-router.md) | P-20260925-balanced-observation-router | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-crossobject-history-value](probes/P-20260925-cm-crossobject-history-value.md) | P-20260925-cm-crossobject-history-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-history-grasp-value](probes/P-20260925-cm-history-grasp-value.md) | P-20260925-cm-history-grasp-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-option-value-retest](probes/P-20260925-cm-option-value-retest.md) | P-20260925-cm-option-value-retest | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-option-value](probes/P-20260925-cm-option-value.md) | P-20260925-cm-option-value | `UNCLEAR` for stable action information, `UNPROMISING` for | 见原卡 | probes/见原卡 |
| [P-20260925-cm-postcontact-heldlift-value](probes/P-20260925-cm-postcontact-heldlift-value.md) | P-20260925-cm-postcontact-heldlift-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-postcontact-three-arm-value](probes/P-20260925-cm-postcontact-three-arm-value.md) | P-20260925-cm-postcontact-three-arm-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-reward-specialist-route](probes/P-20260925-cm-reward-specialist-route.md) | P-20260925-cm-reward-specialist-route | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-route-local-residual](probes/P-20260925-cm-route-local-residual.md) | P-20260925-cm-route-local-residual | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cm-supported-rise-event](probes/P-20260925-cm-supported-rise-event.md) | P-20260925-cm-supported-rise-event | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-contact-expert-option](probes/P-20260925-contact-expert-option.md) | P-20260925-contact-expert-option | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-contact-expert-suffix](probes/P-20260925-contact-expert-suffix.md) | P-20260925-contact-expert-suffix | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cup-route-integration](probes/P-20260925-cup-route-integration.md) | P-20260925-cup-route-integration | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-cup-specialist](probes/P-20260925-cup-specialist.md) | P-20260925-cup-specialist | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-duck-contact-phase-audit](probes/P-20260925-duck-contact-phase-audit.md) | P-20260925-duck-contact-phase-audit | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-expert-choice-headroom](probes/P-20260925-expert-choice-headroom.md) | P-20260925-expert-choice-headroom | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-grab59-extended-baseline](probes/P-20260925-grab59-extended-baseline.md) | P-20260925-grab59-extended-baseline | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-grab59-group-route-cm](probes/P-20260925-grab59-group-route-cm.md) | P-20260925-grab59-group-route-cm | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-grab59-input-gate](probes/P-20260925-grab59-input-gate.md) | P-20260925-grab59-input-gate | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-grab59-shared-baseline](probes/P-20260925-grab59-shared-baseline.md) | P-20260925-grab59-shared-baseline | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-grab660-full-baseline](probes/P-20260925-grab660-full-baseline.md) | P-20260925-grab660-full-baseline | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-observation-route-identifiability](probes/P-20260925-observation-route-identifiability.md) | P-20260925-observation-route-identifiability | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-online-observation-router](probes/P-20260925-online-observation-router.md) | P-20260925-online-observation-router | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260925-six-expert-observation-router](probes/P-20260925-six-expert-observation-router.md) | P-20260925-six-expert-observation-router | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260926-cm-gate-followup](probes/P-20260926-cm-gate-followup.md) | P-20260926-cm-gate-followup | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20260930-cm-physical-value](probes/P-20260930-cm-physical-value.md) | P-20260930-cm-physical-value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-actuation-effect-factorization](probes/P-20261001-actuation-effect-factorization.md) | P-20261001-actuation-effect-factorization | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-contact-response-resolution](probes/P-20261001-contact-response-resolution.md) | P-20261001-contact-response-resolution | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-contact-trigger-geometry](probes/P-20261001-contact-trigger-geometry.md) | P-20261001-contact-trigger-geometry | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-current-policy-v-fit](probes/P-20261001-current-policy-v-fit.md) | P-20261001-current-policy-v-fit | 见原卡 | close HD01 at1/1, retain HF08 PAUSED, do not add updates/seeds to chase | probes/见原卡 |
| [P-20261001-current-policy-value-diagnostic](probes/P-20261001-current-policy-value-diagnostic.md) | P-20261001 current-policy value diagnostic | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-differential-response-learning](probes/P-20261001-differential-response-learning.md) | P-20261001-differential-response-learning | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-direct-randomized-response](probes/P-20261001-direct-randomized-response.md) | P-20261001-direct-randomized-response | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-fresh-causal-transfer](probes/P-20261001-fresh-causal-transfer.md) | P-20261001-fresh-causal-transfer | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-hold-plateau-curriculum](probes/P-20261001-hold-plateau-curriculum.md) | P-20261001-hold-plateau-curriculum | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-hold-plateau-substrate](probes/P-20261001-hold-plateau-substrate.md) | P-20261001-hold-plateau-substrate | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-null-action-recovery](probes/P-20261001-null-action-recovery.md) | P-20261001-null-action-recovery | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-paired-evaluator-resolution](probes/P-20261001-paired-evaluator-resolution.md) | P-20261001-paired-evaluator-resolution | 见原卡 | if noise cannot resolve5pp, close current route without new policy | probes/见原卡 |
| [P-20261001-randomized-effect-risk](probes/P-20261001-randomized-effect-risk.md) | P-20261001-randomized-effect-risk | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-randomized-task-selection](probes/P-20261001-randomized-task-selection.md) | P-20261001-randomized-task-selection | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261001-static-hold-feasibility](probes/P-20261001-static-hold-feasibility.md) | P-20261001-static-hold-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-coherent-barrier-information](probes/P-20261002-coherent-barrier-information.md) | Coherent native execution and task-barrier action information | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-continuous-critic-policy](probes/P-20261002-continuous-critic-policy.md) | Fixed actual continuous PPO comparison | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-delayed-request-response](probes/P-20261002-delayed-request-response.md) | Fixed randomized-request object-response screen | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-empirical-successor-policy](probes/P-20261002-empirical-successor-policy.md) | Actual policy training with a learned observed-successor law | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-finger-preload-feasibility](probes/P-20261002-finger-preload-feasibility.md) | P-20261002-finger-preload-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-force-aware-impulse](probes/P-20261002-force-aware-impulse.md) | Force-aware action-conditioned non-gravity impulse information | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-frame0-tracking-feasibility](probes/P-20261002-frame0-tracking-feasibility.md) | P-20261002-frame0-tracking-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-measured-geometry-barriers](probes/P-20261002-measured-geometry-barriers.md) | Measured object-frame geometry and physical barrier information | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-natural-retention-feedback](probes/P-20261002-natural-retention-feedback.md) | P-20261002-natural-retention-feedback | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-object-relative-transport](probes/P-20261002-object-relative-transport.md) | Object-relative translation opportunity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-observation-hold-aggregation](probes/P-20261002-observation-hold-aggregation.md) | P-20261002-observation-hold-aggregation | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-observation-hold-baseline](probes/P-20261002-observation-hold-baseline.md) | P-20261002-observation-hold-baseline | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-observed-support-task-value](probes/P-20261002-observed-support-task-value.md) | Current observed support information for eventual task value | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-option-model-policy](probes/P-20261002-option-model-policy.md) | Short successor Cm in actual offline option-policy training | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-paired-option-task-opportunity](probes/P-20261002-paired-option-task-opportunity.md) | Paired executed joint-option task opportunity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-physical-encoder-critic](probes/P-20261002-physical-encoder-critic.md) | Actual task critic learning from physical pretrained encoders | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-physical-gradient-control](probes/P-20261002-physical-gradient-control.md) | Fresh measured-return gradient control variate Decision | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-prelift-contact-opportunity](probes/P-20261002-prelift-contact-opportunity.md) | Prelift contact preparation opportunity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-reference-target-policy](probes/P-20261002-reference-target-policy.md) | P-20261002-reference-target-policy | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-request-execution-headroom](probes/P-20261002-request-execution-headroom.md) | Request innovation execution headroom | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-rotation-retention-feasibility](probes/P-20261002-rotation-retention-feasibility.md) | Independent rotation-only retention feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-selective-finger-feasibility](probes/P-20261002-selective-finger-feasibility.md) | P-20261002-selective-finger-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-self-trained-teacher-qualification-goal-alignment](probes/P-20261002-self-trained-teacher-qualification-goal-alignment.md) | Teacher qualification: engineering reference-goal alignment | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-self-trained-teacher-qualification](probes/P-20261002-self-trained-teacher-qualification.md) | Self-trained teacher qualification on current synthetic105task | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-state-response-field](probes/P-20261002-state-response-field.md) | State-conditioned immediate object response: fixed Decision Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-successor-value-compatibility](probes/P-20261002-successor-value-compatibility.md) | Reused eight-step successor/value compatibility Decision | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-support-disturbance-feasibility](probes/P-20261002-support-disturbance-feasibility.md) | P-20261002-support-disturbance-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-support-feature-policy-training](probes/P-20261002-support-feature-policy-training.md) | P-20261002-support-feature-policy-training | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-support-removal-witness](probes/P-20261002-support-removal-witness.md) | P-20261002-support-removal-witness | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-support-response-information](probes/P-20261002-support-response-information.md) | P-20261002-support-response-information | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-task-outcome-determinacy](probes/P-20261002-task-outcome-determinacy.md) | TRAIN-only task outcome determinacy review | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261002-truth-successor-task-value](probes/P-20261002-truth-successor-task-value.md) | Truth-successor eventual task-value upperbound | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-budgeted-physical-critic](probes/P-20261003-budgeted-physical-critic.md) | Budgeted short physical information for actual policy learning | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-cm-granularity](probes/P-20261003-cm-granularity.md) | Fixed-data query and neighborhood granularity Decision Probe | 见原卡 | a >=10% same-hand held-parent gain prioritizes that representation | probes/见原卡 |
| [P-20261003-cm-scale-cross-hand](probes/P-20261003-cm-scale-cross-hand.md) | Cm scale and cross-hand knowledge-prior Decision Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-contrast-acquisition](probes/P-20261003-contrast-acquisition.md) | Fixed-budget physical action-contrast acquisition | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-effect-interaction-oracle-r2](probes/P-20261003-effect-interaction-oracle-r2.md) | P-20261003-effect-interaction-oracle — fixed-background run r2 | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-effect-interaction-oracle](probes/P-20261003-effect-interaction-oracle.md) | P-20261003-effect-interaction-oracle | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-gate1-consequence-value](probes/P-20261003-gate1-consequence-value.md) | P-20261003-gate1-consequence-value — Gate 1 data and bridge readiness | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-inspire-filter-impact](probes/P-20261003-inspire-filter-impact.md) | Corrected shape ownership: fixed-policy sensitivity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-latent-load-feasibility](probes/P-20261003-latent-load-feasibility.md) | P-20261003-latent-load-feasibility | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-oracle-candidate-capacity](probes/P-20261003-oracle-candidate-capacity.md) | P-20261003-oracle-candidate-capacity | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-rigid-coupling-error-decomposition](probes/P-20261003-rigid-coupling-error-decomposition.md) | Fixed-weight coupling error decomposition | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-rigid-coupling-learnability](probes/P-20261003-rigid-coupling-learnability.md) | Learn causal coefficients inside the unchanged transport family | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-rigid-transport-capacity](probes/P-20261003-rigid-transport-capacity.md) | Oracle capacity of an inertial/rigid-transport coupling representation | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-rotational-clearance-adequacy](probes/P-20261003-rotational-clearance-adequacy.md) | Rotation contribution to actual full-mesh clearance decisions | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-state-anchored-transport](probes/P-20261003-state-anchored-transport.md) | State-anchored, proximity-gated direct-flow transport | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-surface-calibration](probes/P-20261003-surface-calibration.md) | Corrected causal surface-feature calibration | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-surface-execution-input](probes/P-20261003-surface-execution-input.md) | Corrected-physics causal execution input qualification | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261003-surface-granularity-audit](probes/P-20261003-surface-granularity-audit.md) | Current geometry coverage audit | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-cmv2-pointflow-chunk-k4](probes/P-20261004-cmv2-pointflow-chunk-k4.md) | P-20261004-cmv2-pointflow-chunk-k4 | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-cmv2-pointflow-single](probes/P-20261004-cmv2-pointflow-single.md) | P-20261004-cmv2-pointflow-single | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-cmv2-unified-ei](probes/P-20261004-cmv2-unified-ei.md) | P-20261004-cmv2-unified-ei | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-gate1-consequence-v2](probes/P-20261004-gate1-consequence-v2.md) | P-20261004-gate1-consequence-v2 — corrected physical timing probe | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-gate1-corrected-coverage](probes/P-20261004-gate1-corrected-coverage.md) | P-20261004-gate1-corrected-coverage | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-gate1-split-rng](probes/P-20261004-gate1-split-rng.md) | P-20261004-gate1-split-rng | with the original data and fixed splits, changing only the model seed | if split 3 stays negative and split 1 stays positive across the same | probes/见原卡 |
| [P-20261004-gate2-action-chunk-k8](probes/P-20261004-gate2-action-chunk-k8.md) | P-20261004 Gate 2 action chunk K=8 | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-gate2-predictability](probes/P-20261004-gate2-predictability.md) | P-20261004 Gate 2 consequence predictability | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-ref4-g-bridge](probes/P-20261004-ref4-g-bridge.md) | P-20261004 Ref4 G bridge | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-ref4-g-predicted-e](probes/P-20261004-ref4-g-predicted-e.md) | P-20261004 Ref4 predicted-E bridge | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261004-surface-token-i](probes/P-20261004-surface-token-i.md) | P-20261004-surface-token-i | 见原卡 | 见原卡 | probes/见原卡 |
| [PROBE-20260923-CM-EFFECT-ACTION-ALIGNMENT](probes/PROBE-20260923-CM-EFFECT-ACTION-ALIGNMENT.md) | Cm 效应头的动作对应关系 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [PROBE-20260923-CM-WEIGHT-COMPONENTS](probes/PROBE-20260923-CM-WEIGHT-COMPONENTS.md) | Cm PPO 权重分量离线 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [PROBE-20260923-CM-WEIGHT-ONLINE-HEADS](probes/PROBE-20260923-CM-WEIGHT-ONLINE-HEADS.md) | Cm PPO 权重分量在线 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261008-pointworld-ref8-voxel-static](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261008-pointworld-ref8-voxel-static.md) | P-20261008-pointworld-ref8-voxel-static | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261009-pointworld-main3-sharedstats](../../src/task/cm-pointflow-effect-pretrain/docs/experiments/probes/P-20261009-pointworld-main3-sharedstats.md) | P-20261009-pointworld-main3-sharedstats | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261009-gpu-group-contract](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-contract.md) | Native GPU group role/layout contract | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261009-gpu-group-value-noise](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-group-value-noise.md) | Native GPU synchronous group Value-noise calibration | 见原卡 | 见原卡 | probes/见原卡 |
| [P-20261009-gpu-pipeline-threads1](../../src/task/consequence-evaluator/docs/experiments/probes/P-20261009-gpu-pipeline-threads1.md) | GPU pipeline with one PhysX task thread | 见原卡 | 见原卡 | probes/见原卡 |
| [VAL-20260923-CM-EFFECT-PPO](validations/VAL-20260923-CM-EFFECT-PPO.md) | Validation: 动作条件 Cm 效应排序能否稳定改善 PPO？ | 见原卡 | 见原卡 | validations/见原卡 |
