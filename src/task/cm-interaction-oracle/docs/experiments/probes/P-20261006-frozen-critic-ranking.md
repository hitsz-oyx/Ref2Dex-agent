---
schema: ref2dex.probe.v2
probe_id: P-20261006-frozen-critic-ranking
experiment_id: P-20261006-frozen-critic-ranking
date: 2026-10-06
task: cm-interaction-oracle
branch: agent/cm-interaction-oracle
git_commit: b94686556267249ddf402e3f66fbf38af9340f22
claim_id: C3
hypothesis_family: HF-frozen-critic-ranking
probe_index_in_family: 1
seed_pool: probe
seeds: [263, 264, 281]
decision_changed_if_positive: qualify a separately frozen real rolling critic comparison against GT-Y
decision_changed_if_negative: retain Y and do not feed Cm predicted states into this frozen critic
status: PLANNED
run_id: frozen-critic-ranking-source-e260
---

# Can frozen PPO critic replace short-Y candidate ranking?

## Root Decision Note / Decision experiment

User explicitly requests critic ranking using ref14 baseline source_e260 PPO
checkpoint, superseding the earlier pause on critic experiments for this bounded
diagnostic. Ref14_4 motivates GT future state + frozen V, with no Cm training.
Hypothesis: existing policy-dependent V and PPO reward can identify useful local
candidates at least as well as GT-Y. Cheapest adequate test uses all32original
same-prefix anchors/seven candidates with known90step Z. Raw future critic
observations1442 and eight rewards were NOT saved, so a short cold replay must
capture them; physical72/q alone cannot reconstruct the critic input contract.
No new resetcohort, candidate, actor, reward, Y or Z contract. No policy/model fit.

## Frozen checkpoint and score

`source_e260`: `outputs/Dexplore/agent_v139_s3_backtrack_s70_e260/train/inspire_slow_slow_energy_reset_contact_table_adjust_parameter_2/nn/GRAB_00000260.pth`
in the existing baseline output root; SHA256
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.
Checkpoint critic1024/1024/1024/512 ReLU, same raw1442D obs as actor, external
observation RMS. Normalize exactly with checkpoint mean/variance, epsilon1e-5,
clamp[-5,5]. Source training config: gamma=.99, reward_shaper.scale_value=1,
normalize_value=false; no value de-normalization. Critic and actor stay frozen.

Primary score `S=sum(j=0..7,gamma^j*r[t+j])+gamma^8*V(s[t+8])`.
The actual eight feedback actions and resulting native future observation are
used, including intervention. Capture each environment reward AFTER its action;
endpoint AFTER the eighth action, before a reset/next action. Abort if a done
occurs in that block; do not bootstrap a reset observation. Reward-only and
V-only rankings are predeclared diagnostics, not post-hoc replacements for S.
Preserve source reward units, reference/time features and baseline-first ties.
Native PPO reward reflects imitation/energy/contact terms, not necessarily Z;
V is policy-dependent. No Cm/predicted future state enters this experiment.

## Paired data, checks and evaluation

Read-only original ref13 four synchronous solver groups,32current-only anchors,
seven candidates and unchanged32step Y/90step Z. Source runs:
`oracle-y-utility-s263-sync-reference`, `oracle-y-utility-extra-s264-sync-reference`,
their existing schedules and `*-sync-g<group>-candidate<k>` panels.
Capture28original-mode90step candidate replays, including all four group baselines.
Match currentH, actions/PD, fullphysical32/q/tips/base<=1e-4; height/pair/valid90
must replay EXACT. Inherited full-world prefix<=1e-4 and shadowactor<=1e-5.
Never infer new90step Z from short replay: use the corresponding original
candidate90, joined by seed/group/row/candidate, only after matching first32.
Source Z23/32, GT-Y24/32 and finite candidateupper25/32 must reconstruct.
Live player critic must agree with independent checkpoint reconstruction<=1e-5;
actor/model/RMS fingerprints must remain unchanged. Save raw endpoint obs/reward,
original source hashes, raw values, reward/bootstrap components and score.

Report candidate-selected Z count, rescue/harm, selections, paired bootstrap
vs baseline/GT-Y, every rescue opportunity, Z-discordant pair ranking and strict
Y-ranking agreement; ties explicitly counted. Counts are a potential-outcome
mosaic from synchronous groups, NOT actual mixed-action or rolling-policy Z.
Do not compare this one-shot counterfactual mosaic as if it were the already
executed rolling GT-Y27/32. Report that reference with its separate scope.
Use anchorbootstrap2000seed281, descriptive only, not solver independence.

Offline adequacy>=30anchors/>=2motions. PROMISING for advancing to a new rolling
Probe if primary S selects at least as many successful Z as same-panel GT-Y,
rescues>=1 and harms=0; otherwise UNPROMISING for this fixed score, UNCLEAR if
coverage is insufficient. This is a screening gate, not supported replacement
of Y. No MSE threshold or need to imitate Y perfectly: Z is the common objective.
If negative, inspect score decomposition/actual input execution before closure;
no sign flip, coeff search, new critic fitting or frozen-score retuning.

## Resource and stop boundaries

SameGPU PhysX/CPU tensor pipeline, GPU actor/critic. Smoke onebaseline group:
<=240s/0.25GiB; main<=1200s/2GiB, two idleGPU1/2, one worker perGPU,2concurrent.
Each worker180s internal/240s process cap. Saved-weight audit<=120s GPU.
Preflight freeGPU/disk; existing outputs/checkpoints remain immutable. New output
`outputs/cm-interaction-oracle/frozen-critic-ranking-source-e260/`; fixedcode
commit/run identity and protocol hashes before main. Stop on replay/input/hash/
label/normalization drift, nonfinite, terminal, resource conflict or cap.

## Limitations / future evidence

Exposed32anchors/fourshared solver groups and finite candidates constrain all
counts. Exact candidate replay is not a real composed selector-policy run.
Even a positive screen needs fresh same-current-state rolling critic execution
vs matched baseline/GT-Y before replacement claims. Prospective Cm→predicted
state→V and final matched trained-policy Cm-on/off remain separate questions.

## Engineering smoke and frozen main choice

Smoke r3 at42c5731 passes:7anchor first synchronous group, currentH/history,
first8actions/PD and physical32/q/tips/base all EXACT against old baseline;
height/pair/valid32 EXACT; live and independent checkpoint critic error0.
Elapsed28.69s. Earlier startup failures retained in smoke/r2 folders (relative
worker path and reward-shaper object API), repaired before physics/scientific
results; not negative method evidence. Combined smoke elapsed<240s.
GPU0 now belongs to another process; main uses idleGPU1/2, one worker each.
112Task tests pass,1skip. Main code/config/protocol hashes fixed before launch.

## Replay-contract correction before scientific completion

Original main `frozen-critic-ranking-source-e260` atdb0d9ab FAILED after125.08s;
5completion events, no final scientific result. s263/g0/grip+ first8physical and
actions EXACT, but poststep11microdifference and step18+physical divergence;
height max.004599m, q.09873rad,3pair discrepancies, Y/U differs. Root and
independent review identify that --rolling-offset0 changed postfork RNG replay:
oldref13 restores reference RNG throughout90; rolling stops afterfork. Old90Z
cannot be joined under the registered first32gate. Preserve all failed outputs;
none are negative critic evidence. Do not relax thresholds or overwrite data.

Restore original nonrolling90step collection; capture only eight rewards and
endpointV, verify physical32 and entire height/pair/valid90 exactly as above.
Use failing grip+ as the one-variable engineering regression (sameGPU2), then
new run_id `frozen-critic-ranking-source-e260-r2` for complete repaired capture.
Prior125.09seconds charged to original1200main cap, leaving1074.91seconds;
engineering attempts stay within shared240second smoke cap. No score, candidate,
label or checkpoint changes. Independent review of repaired outputs precedes
method interpretation. Model/optimizer source and PPO reward remain unchanged.


Regression `frozen-critic-ranking-grip-regression` atd8ead1e completes37.26s:
same failing candidate5/group0/GPU2, entire physical32/q/tips/base/action/currentH
EXACT, fullheight/pair/valid90 EXACT, critic replay error0. This isolates restored
nonrolling RNG semantics as the effective fix, with no relaxed match gate.
Original failed scores remain excluded; repaired main uses all28fresh candidate
replays and source90labels, no model fit or extra candidate/resets. Main budget
passes `--spent-seconds125.09` and uses original remaining1074.91seconds.
