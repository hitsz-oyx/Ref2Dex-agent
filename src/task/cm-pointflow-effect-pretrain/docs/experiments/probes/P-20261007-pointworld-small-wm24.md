---
schema: ref2dex.probe.v2
probe_id: P-20261007-pointworld-small-wm24
experiment_id: P-20261007-pointworld-small-wm24
date: 2026-10-07
task: cm-pointflow-effect-pretrain
branch: cm-pointflow-effect-pretrain
git_commit: f507f18
claim_id: C3
hypothesis_family: HF-pointworld-unified-action-effect
probe_index_in_family: 1
seed_pool: probe
seeds: [210, 212, 213, 214, 215, 216, 217, 218]
decision_changed_if_positive: continue unified spatial effect pretraining and design robot adaptation
decision_changed_if_negative: inspect learning and action use before additional training
status: UNCLEAR
run_id: pointworld-small-wm24-20261007
---

# Unified spatial hand/action-to-rigid-effect Probe

Decision experiment serving Mission short-term physical prediction: distinguish
whether the ref3 unified PointWorld architecture and normalized motion losses
learn useful observed-hand information rather than static predictions. Cheapest
next gate is train-only stats plus real-data engineering/learning smoke, followed
by bounded three-arm training on the already prepared corpus.

## Decision Note

Old WM30 outputs near-zero despite moving labels and action gradients; this
does not refute Cm. User explicitly requests ref3 replacement. Choose the
specified PTv3-small backbone, delete separate action/dynamics Transformers,
retain exact rigid GT and implement released normalization/soft weighting.
Existing branch; corpus reused without downloads; at most4 GPUs overall,
24h group wall budget/40000updates, new outputs only. Preserve old runs and,
when needed, save/stop identified old workers. Stop on nonfinite, code/input
drift, disk conflict, deadline. Positive screen leads to adaptation design;
negative/uncertain signal leads to attribution rather than extra seeds/epochs.
No external authorization boundary is crossed and Mission claim stays unchanged.

## Frozen design and implementation

[User ref3](../../user/ref/ref3.md) and
[concrete design/implementation sequence](../../POINTWORLD_WM24_DESIGN.md).
Run output: `outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/`.
Input: completed `outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.
Actual runtime commit and source/stat/config hashes are authoritative in manifests;
frontmatter commit is the verified implementation checkpoint.

## Evaluation / decision

Same H/H+A/H+shuffle(A), initialization, draw schedule,40000updates and effective
batch16; balanced/natural val256 and held-out test256, sequence isolation. B also
evaluated with shuffled A. Report physical pointEPE/translation/rotation/center
error at h1/4/8/12/24 for anchor,moving objects,scene and each stratum, with static
baseline. PROMISING requires moving-anchor h24 test EPE>=10% lower than BOTH
controls, beats static, and inference shuffle degrades>=5%. Otherwise valid
equal-update failure is UNPROMISING, execution/information insufficiency UNCLEAR.
Engineering smoke is only interface/learnability evidence, not the test screen.

## Current status

Implementation and real-GPU engineering gate complete:50,495,881 parameters,
7tests passed/1optional old-model GPU test skipped. Three6-update independent
arms share identical initial weights and exact checkpoint roundtrip. Resume
loads and standalone validation inference pass. Corrected80-update repeated
batch loss3.01555→0.22490 and anchor EPE44.335→16.382mm show plumbing/fit only.
Train-only4096-window normalization is frozen; old workers saved/stopped by
explicit user request. Full three-arm training launched on GPUs0/1/2 at
runtime commit `f507f18`;launcher PID3942111 and workers3942116/3942117/3942118.
All arms have completed initial updates with finite losses/gradients. Actual
normalization/config/data/initial parameter hashes match across the three arms.
Deadline is the shared24h timestamp in group_status.json; first observed steady
updates take approximately1.2–1.4s,roughly14–16h for40000 before longer-run
throughput is established. No held-out predictive conclusion yet: UNCLEAR.
Runtime identity/progress checks are saved in `startup_verified.json`.

Current-route `VERIFY_BASE=ce6fcb4 tools/verify.py --changed` passes, including
schema/links/seed checks and selected governance tests. Default whole-branch
verification includes restored history and reports102passed/6failed; those six
legacy tests require deleted baseline checkpoint/data artifacts. Representative
preflight fails on absent `Ref2Dex-agent-baseline/outputs/CmResidual/agent_contact_option_airplane_motions`.
These are the already recorded recovery limitation, not a ref3 dependency.

During engineering, duplicate sparse coordinates caused nondeterministic CPE
outputs; corrected with unique unified1cm voxels and inverse restoration, plus
fixed eval serialization at every pooling stage. Failed logs and early smoke
remain; invalid adapter outputs do not count as negative ref3 evidence.
[Full interface record](../../POINTWORLD_INTERFACE.md).

Final reporting now has a read-only Task audit entry,
`tools/audit/summarize_pointworld_screen.py`. It holds the numeric screen at
NOT_READY until equal-update final TEST evaluation is complete, checks original
identities/checkpoint hashes and replays the fixed test static label panel on
CPU. Gate arithmetic tests include exact10%/5% boundaries, nonfinite errors,
unequal updates and missing test output. While training is live it checks the
own process command lines and reads progress only; no test panel is opened.
The first pending snapshot confirms all four processes live after~230updates.
This adds final-result audit readiness; it does not change the active training
code/config/data, protocol, resource budget or current UNCLEAR conclusion.

## First matched validation: update 1000

All three arms reached the same first validation checkpoint. The fixed balanced
256-window validation panel gives the following moving-anchor (cat0) point EPE;
distances are millimeters. This is intermediate validation, not the held-out
final test screen.

| Arm | h1 | h4 | h8 | h12 | h24 (0.8s) | Mean training loss, updates 901–1000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| History | 1.12 | 4.08 | 8.19 | 12.41 | 25.51 | 2.3383 |
| History + future hands | 1.41 | 4.86 | 9.03 | 13.44 | 26.31 | 2.2853 |
| History + shuffled future hands | 1.21 | 4.11 | 8.25 | 11.99 | 23.99 | 2.4097 |
| Static prediction | 1.69 | 5.95 | 11.04 | 15.33 | 26.13 | — |

Training loss is dimensionless and normalized; a lower training loss is not a
lower physical prediction error. Short-horizon predictions beat static in this
panel, but at h24 the true future-hand arm does not yet beat either control or
static. Current judgment remains UNCLEAR: this early checkpoint does not
establish an action benefit or close the route. Continue the frozen 40000-update
run without changing active sources/configuration or extending the budget.

The complete first-validation metrics and live-process evidence are preserved
in `outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/first_validation_observed.json`;
the mean losses are from each arm's `train.jsonl`, updates 901–1000 inclusive.
The upstream PointWorld source is now also recorded as a proper Git submodule
at the already-used commit; this repository bookkeeping does not alter any
running source bytes or the original runtime identity `f507f18`.

## Limitations / future evidence

Observed future hands are post-treatment geometry, not counterfactual commands.
One-seed observational accuracy cannot establish causal robot/PPO utility or
global C3. Later multi-seed matched validation, robot adaptation and trained
Cm-on/off comparison remain necessary. PTv3 and loss change together; old/new
differences cannot separately identify architectural contribution. Sample-based
train normalization is approximate; held-out statistics are never fitted.
