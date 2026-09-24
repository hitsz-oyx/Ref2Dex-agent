# P-20260924-multiaxis-h10-online-choice

date: 2026-09-24
branch: `agent/cm-multiaxis-long-horizon`
classification: Decision

## Question and decision

Can the frozen, newly trained multiaxis/ten-step Cm improve complete
held-lift by choosing between contact-preserving wrist x+ and
lift-progress wrist z+, rather than applying either blindly?
If not, stop this fixed short-horizon online selector; do not tune
thresholds on observed evaluation seeds.

## Frozen selector and controls

Use the self-trained e260 actor, corrected s3 motion, and frozen
`raw_action.pt` from `P-20260924-multiaxis-h10-cm`. Raw is selected
before policy evaluation because on held-out seed163 it had lower
contact RMSE and substantially lower y/z displacement RMSE than the
geometric model, while also passing the pre-set x-effect gate.
No model retraining or online adaptation.

In actual contact and no reset, only at global steps 50..200 stride10,
score the base action, wrist x+0.1 and wrist z+0.1. Choose z+ if
predicted ten-step object z gains ≥5 mm and ten-step contact fraction
falls ≤2 pp versus base; otherwise choose x+ if predicted contact
gains ≥3 pp and object z loses ≤5 mm; otherwise keep base.
The z+ priority and thresholds are fixed from pre-evaluation physical
effect scale, not optimized on new policy seeds. Require action dose
unclipped. Each model query uses only current q, dof velocity, object
state and candidate action; no future state/label.

Arms: `base`, `always_x`, `always_z`, `cm`. Blind arms apply their
single +0.1 dose to every same eligible contact/time window; this
distinguishes model choice from generic intervention. All four use
the identical frozen actor and simulator setup. First run a 16-env
`cm` engineering smoke. Then evaluate seed164 (64 env, full first
episode) for all four arms; if `cm` is below `base`, stop early since
the per-seed nonnegative gate is already impossible. Otherwise repeat
the four arms at seed165.

Primary metric: strict full-first-episode held-lift successes. Upgrade
to matched multi-seed/placebo Validation only if Cm beats base and
both always arms by ≥8 percentage points overall, is nonnegative
versus base in each evaluation seed, and actually selects both x+ and
z+ at least 20 times pooled. Reward, contact and lift trajectory are
diagnostic only. An engineering smoke or single seed cannot prove
policy utility.

## Budget and stop

Each run one idle GPU and ≤64 env, total ≤2 GPUs concurrent, all runs
<60 min and <250 MB outputs. Stop on actor/model/motion SHA drift,
GPU conflict, non-finite scores, clipped dose, incomplete evaluation,
or budget breach. No training checkpoint modification.

## Result

Status: PENDING

Evidence: pending

## Decision update

pending
