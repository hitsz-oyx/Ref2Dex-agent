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

## HF-amplitude-authority

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-amplitude-authority](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-amplitude-authority.md) | Is there a task-related interaction threshold within larger feasible actions? | Task-aligned authority UNPROMISING; localized thumb interaction response PROMISING (Probe only). | Preserve local thumb control evidence; no registered task-aligned threshold, so conditional Stage2 and neural fitting are not activated. | probes/COMPLETED |

## HF-conditional-consequence

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-conditional-consequence](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-conditional-consequence.md) | Can predicted action consequences retain oracle task information? | Pending fixed conditional-predictability and OOF task-transfer Probe. | Test action contrasts and value against direct Ha before candidate/policy integration. | probes/PLANNED |

## HF-gt-consequence-sufficiency

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-gt-consequence-sufficiency](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-gt-consequence-sufficiency.md) | Do GT physical consequences preserve task-relevant action information? | GT prognosis PROMISING (46.25% primary error gain); held-arm EI bridge positive; full sufficiency UNCLEAR because remaining-action uncertain | Preserve task-relevant E/I evidence and qualify the next conditional-predictability question; no claim of identified sufficiency, online sel | probes/UNCLEAR |

## HF-hold-duration-response

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-early-hold-duration](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-early-hold-duration.md) | Does longer feedback-residual execution change grasp-retention information? | 1006 full randomized windows; dose and support pass, but no duration-sensitive retention-I gate; hand displacement responds while I16/contac | Close K4/8/16 feedback-residual extension after independent review; no Cm fitting, extra duration/seed or selector. | probes/UNPROMISING |

## HF-per-finger-control

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261005-per-finger-control](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261005-per-finger-control.md) | Does separating the finger synergy expose task-related interaction control? | 854 full windows; registered linked/useful gates UNPROMISING, with late-contact action response and yaw-minus/middle-minus clues retained. | Preserve physical magnitude tables and targeted task clues; no amplitude/seed sweep or Cm fitting from this completed contract. | probes/COMPLETED |

## HF-pointflow-G

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-pointflow-g](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-pointflow-g.md) | Does real point-flow predicted E preserve E-to-G information? | UNCLEAR for the valid K1 point-flow-to-G bridge; UNPROMISING for expanding a pose-only K4 predictor into this G regression from the fixed GT | keep E as the physical main route, freeze I head/K4 expansion and this frozen K1 G teacher; audit task-relevant hold/drop value evidence bef | probes/UNCLEAR |

## HF-relative-action

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-recap-relative-action](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-recap-relative-action.md) | Can relative task outcome supervise a deployable action critic? | G0 label stability failed (train34.7%, test39.1%); counts/support and V-versus-zero checks pass; G1 not executed. | retain the relative-outcome direction but stop critic/Cm expansion on this unstable discrete label contract; no action-information or policy | probes/UNCLEAR |
| [P-20261004-relative-action-ranking](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-relative-action-ranking.md) | Does current action improve ranking of frozen relative task advantage? | H macro pair59.71%, HaK1 59.62%, HaK4 realized60.52%; current-action ranking gate fails after independent implementation review. | stop critic/Cm expansion on this frozen target/data/fit contract; proxy ordering also lacks strong V-fit stability, so do not refute physica | probes/UNPROMISING |

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

## REF5-SURFACE-I

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20261004-ref5-surface-i-gt-value](../../src/task/cm-interaction-oracle/docs/experiments/probes/P-20261004-ref5-surface-i-gt-value.md) | Ref5: does GT spatial surface I add G information after H and E? | 见原卡 | 见原卡 | probes/UNPROMISING |

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
| [VAL-20260923-CM-EFFECT-PPO](validations/VAL-20260923-CM-EFFECT-PPO.md) | Validation: 动作条件 Cm 效应排序能否稳定改善 PPO？ | 见原卡 | 见原卡 | validations/见原卡 |
