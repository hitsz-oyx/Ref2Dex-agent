# 实验阅读索引

由 tools/experiment_index.py 生成；状态照录原卡，不重判科研结论。
问题、结果与下一步见原卡开头；详细配置和执行版本见其 manifest。

## HB01

| 实验 | 问题 | 结果摘要 | 下一步 | 类型/状态 |
| --- | --- | --- | --- | --- |
| [P-20260926-observation-router-reliability](probes/P-20260926-observation-router-reliability.md) | New-seed reliability of the observation-driven six-expert baseline | 见原卡 | 见原卡 | probes/PROMISING |
| [VAL-20260926-observation-six-expert-c1](validations/VAL-20260926-observation-six-expert-c1.md) | Validation: frozen observation-driven six-expert C1 route | 见原卡 | 见原卡 | validations/COMPLETED |

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
| [PROBE-20260923-CM-EFFECT-ACTION-ALIGNMENT](probes/PROBE-20260923-CM-EFFECT-ACTION-ALIGNMENT.md) | Cm 效应头的动作对应关系 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [PROBE-20260923-CM-WEIGHT-COMPONENTS](probes/PROBE-20260923-CM-WEIGHT-COMPONENTS.md) | Cm PPO 权重分量离线 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [PROBE-20260923-CM-WEIGHT-ONLINE-HEADS](probes/PROBE-20260923-CM-WEIGHT-ONLINE-HEADS.md) | Cm PPO 权重分量在线 Probe | 见原卡 | 见原卡 | probes/见原卡 |
| [VAL-20260923-CM-EFFECT-PPO](validations/VAL-20260923-CM-EFFECT-PPO.md) | Validation: 动作条件 Cm 效应排序能否稳定改善 PPO？ | 见原卡 | 见原卡 | validations/见原卡 |
