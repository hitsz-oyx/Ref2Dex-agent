# P-20261004-gate1-corrected-coverage

Classification: **Decision Probe**. Status: `PROMISING`, not Validation.

After repairing the factual preceding-action history contract, four new runs were
collected from the pinned e260 checkpoint with the same pre-step collector
(namespaces 1866--1869). The new runs contain 56 episodes and 60,756 rows; one
episode is `stable_success` and none are `drop_after_success`. Merging with the
previous four corrected runs gives 224 episodes and 116,067 h16 windows.

All transforms and audits passed. The corrected direct I+ h16 results are
`+12.6%`, `+5.2%`, `+18.3%`, `+26.9%`, `+14.9%`; the five future-action controls
are `+15.9%`, `+18.9%`, `+25.3%`, `+25.3%`, `+40.7%`. Direct episode-bootstrap
CI excludes zero in 4/5 splits; control excludes zero in 5/5. Eight-source-run
cluster bootstrap leaves only direct split 1 crossing zero, with all control
intervals positive.

The strict action-inclusive `V_HAEI` relative to `V_H` is
`+13.7%`, `+9.8%`, `-3.2%`, `+24.8%`, `+18.2%`; its split 3 interval still
crosses zero. Thus the control evidence is encouraging but does not erase the
remaining action/outcome sensitivity.

This is evidence that outcome coverage reduced the prior split-3 influence, not a
formal causal result: data expansion was selected after exploratory diagnostics,
success/drop coverage remains imbalanced, and the independent e420 actor did not
replicate the direction. Preserve the artifacts and design a pre-registered
actor/outcome-cluster Validation before any Cm training.

Artifacts: `tmp/e260_all8_h16_histfix.pt`,
`tmp/e260_all8_h16_histfix_i_aug.pt`, the corresponding audit JSONs, and
`tmp/gate1_split_rng/e260_all8_histfix_i_aug_fit_h16_[1-5].json` plus controls.
