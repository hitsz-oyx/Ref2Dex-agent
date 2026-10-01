# P-20261001-targeted-contact-interventions

HF09 slot3/3, Decision Probe; frozen proposer and fresh randomized data repair.
Design/budget/decision rules fixed before execution in
[decision memo](../../decisions/D-20261001-targeted-contact-interventions.md).

Hypotheses: Cm's proposed deviations have actionable local lift benefit versus
base on actual states, or its factual prediction improvement did not transfer
to effect ranking. Slot2's low intervention matching cannot resolve this.
Cheapest informative action: concentrate1:1 randomization at these proposals,
without fitting a new ranker, modifying the old held gate, or training PPO.

Frozen checkpoint: ranking-r1/model_fit-r1/ranker.pt,
SHA256 a38be701e1d32fbe66ea4c4ecc4e78178d205f5d8649beb921818ace364e808d.
Same six self-trained experts/three airplane motions/base4; contact force proxy,
history10, candidate2+base8, cooldown6,96env, max8 windows/first episode,
max650ticks. New simulator341–346, private allocation7341–7346. Assignment
is after state/proposal observation; p=.5 among active proposed interventions,
p=1 base elsewhere. All started windows must complete, no clipping/trimming
actual labels. Model/calibration remain fixed; no old held targets are inputs.

Scope: actual receding observation/proposals in randomized mixed history.
Report action changes, repeated decisions, conditional active lift/contact/drop
effect with episode and initial-frame clustering, inference latency. No
individual oracle/regret or full pure-policy/terminal success claim. Screen
needs≥128active windows,≥48windows/24episodes in each arm. Risk requires
≥20lifted windows/3events per arm; otherwise safety UNCLEAR. For positive local
lift:≥.5mm and descriptive frame-group90% lower>0; contact≥−.05/drop≤+.05.
These are Probe criteria, not formal validation. Unsupported heads/events stay
unsupported even with successful action wiring. Unique owned outputs, total
≤60min/8GiB including failed engineering attempts; GPU first. Stop on
input drift, unfinished windows, frozen-model changes, or resource violation.
