# P-20261004-gate1-corrected-coverage

Classification: **Decision Probe**. Status: `PROMISING`, not Validation.

After repairing the factual preceding-action history contract, four new runs were
collected from the pinned e260 checkpoint with the same pre-step collector
(namespaces 1866--1869). The new runs contain 56 episodes and 60,756 rows; one
episode is `stable_success` and none are `drop_after_success`. Merging with the
previous four corrected runs gives 224 episodes and 116,067 h16 windows.

All transforms and audits passed. The corrected direct I+ h16 results are
`+12.6%`, `+5.2%`, `+18.3%`, `+26.9%`, `+14.9%`; after repairing the fast
assembler's episode-local future-action index, the five controls are
`+19.1%`, `-0.6%`, `+14.0%`, `+14.8%`, `+15.3%`. Direct episode-bootstrap CI
excludes zero in 4/5 splits; the repaired control excludes zero in 4/5. Eight-
source-run cluster bootstrap leaves only direct split 1 and control split 2
crossing zero.

The strict action-inclusive `V_HAEI` relative to `V_H` is
`+13.7%`, `+9.8%`, `-3.2%`, `+24.8%`, `+18.2%`; its split 3 interval still
crosses zero. Thus the control evidence is encouraging but does not erase the
remaining action/outcome sensitivity.

This is evidence that outcome coverage reduced the prior split-3 influence, not a
formal causal result: data expansion was selected after exploratory diagnostics,
success/drop coverage remains imbalanced, and the independent e420 actor did not
replicate the direction. Preserve the artifacts and design a pre-registered
actor/outcome-cluster Validation before any Cm training.

The follow-up corrected horizon probe fixed the next Validation candidate at
`H=32`: after deterministic quaternion-sign canonicalization, direct I+ across five
seeds is `+7.7%`, `+24.7%`, `+34.9%`, `+45.9%`, `+35.6%`; episode CI excludes zero
in 4/5 and source-run cluster CI in 4/5. The repaired future-action control is
`+21.2%`, `+13.8%`, `+17.4%`, `+35.3%`, `+33.1%`, positive in every episode and
cluster CI. H3 and h5 remain seed-sensitive, while h10 direct is only
`+5.1%` in the screening seed (its action-inclusive arm is `+21.6%`).

Artifacts: `tmp/e260_all8_h16_histfix_fastfix_i_aug.pt`, its base/augmentation audit
JSONs, and `tmp/gate1_split_rng/e260_all8_histfix_fastfix_h16_ctrl_2026122*.json`
plus the corresponding actor-cluster bootstrap JSON. The direct I+ artifacts remain
the previously audited `e260_all8_histfix_i_aug` outputs; the repaired control is the
only future-action evidence used here.
