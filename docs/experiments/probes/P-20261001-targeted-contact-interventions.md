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

## Run r1

Code dbe3212317c86a08667f2926dd46395535c69959;15 targeted checks pass,
including real runtime normalization/RNG preservation and sequential proposal/
base allocation, exact2step execution, and actual-state re-observation. Small
isolated unit tests use CPU; model inference and real simulation use admitted
GPU4 UUID GPU-0606f00a-d9d0-3a00-5b49-c9e747b77307.

run_status: COMPLETED. Owned output:
`src/task/CmResidual/research/contact_consequence/output/P-20261001-targeted-contact-interventions-r1/`
with `run_manifest.json`, per-seed logs/manifests and complete `records.pt`.

```sh
python3 -u scripts/run_randomized_contact_collection.py --output src/task/CmResidual/research/contact_consequence/output/P-20261001-targeted-contact-interventions-r1 --gpu 4 --ranker src/task/CmResidual/research/contact_consequence/output/P-20261001-contact-consequence-ranking-r1/model_fit-r1/ranker.pt
```

No result or benefit conclusion at launch. All six fixed phases required,
even if intermediate outcome estimates are disappointing.

## Terminal results and route decision

Six phases completed,3941windows/547episodes;622proposals randomized309Cm/
313base, from183/195episodes.534episodes had repeated actual observations
and decisions. Proposal15.8%, actual changed actions7.84% under1:1 allocation;
nonbase action L2difference median.088. Inference batch median5.71ms/p90 6.04ms
(max36.66ms). Single GPU4 collection563.96s,28.2MB; original inputs hash-verified
at completion, all own native PIDs exited and GPU released.15 targeted tests.

Original predeclared local screen PROMISING, all arithmetic gates pass:
active supported-lift effect+5.106mm, descriptive frame-cluster95%interval
[+1.361,+8.850]mm; all-window local effect+.806mm[+.138,+1.474]. This is
prospective randomized mixed-history decision evidence; no individual oracle
or pure-policy/long-term success claim. Old-held-frame descriptive subgroup
has130active windows and+9.999mm signal, so the whole signal is not confined
to frames used for fitting. Subgroups were not used to tune a model or gate.

Safety and attribution prevent utility/training promotion: eligible-drop
counts11/140Cm vs4/122base; known-propensity eligible-drop effect+5.344pp,
frame95%[−1.956,+12.643]pp. All-active drop+2.251pp passes the declared5pp
arithmetic guard but dilutes already-lifted risk. Retained supported lift
(minimum height in last3steps with contact throughout those3) effect+2.901mm,
frame95%[−2.104,+7.906], weaker than mean-positive lift.35.1% of Cm's positive
lift windows lose that retained progress, versus26.8%base;7Cm/2base windows
drop despite positive mean-lift labels. This does not prove a risk increase or
that all gains are transient, but makes the physical objective/safety repair
necessary. Always-base is the only supported decision contrast here; original
slot2 four-control failure remains, no claim of numeric-Cm-specific utility.

Scientific scope: local mean supported lift PROMISING, overall safe policy
utility UNCLEAR. Preserve the original gate, checkpoint and all results.
HF09 closes its3/3slots with useful information. Next mechanism work should
predict/score retained physical trajectories and action-relative effects,
not increase PPO supervision or silently loosen thresholds. Detailed frozen
results and terminal audit: [results](P-20261001-targeted-contact-interventions-results.json).
