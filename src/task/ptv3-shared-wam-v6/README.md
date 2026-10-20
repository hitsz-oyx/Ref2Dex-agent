# WAM v6 — object goal plus next CONTACT/RELEASE event

The user created this task to separate event-conditioned work from the v5
baseline and make rollback a directory/checkpoint choice. The v6 package,
model implementation, training launcher, data recipe and checkpoint schema
are independent. v5 production entrypoints and configs are unchanged; its
historical optional event prototype remains available for audit compatibility.
Shared v5 geometry, losses, metrics, normalization and frozen data utilities
are reused from the same archived code snapshot, not a mutable remote checkout.

Inputs are H4 observed geometry, the original object endpoint goal and one
planned NONE/CONTACT/RELEASE event with relative30Hz time/validity. The zero
initialized event residual adds50304 parameters without new tokens, points or
KNN. Joint21/K24,2048 object/C queries,778 samples per hand,5mm PTv3 and original
TRAIN-only horizon normalization remain unchanged.

Current status (2026-10-20): stopped after GPU3 reported requires reset;
latest complete checkpoint17000 is retained. Administrator GPU recovery is
needed before strict four-GPU resume. The stale progress RUNNING is not live.

Historical startup (2026-10-20): the user chose continuing v6 after the readiness
check. v5 saved/stopped at15723; v6 strict-resumed its own16136 checkpoint into
pretrain-nine-v6-events-20261020-r1. See the
[current training and target audit](docs/experiments/probes/P-20261020-wam-v6-mixed-pretrain.md).
Sparse inherited Shadow runaway targets are diagnosed; concrete cleanup
candidates are recorded separately and have not changed the frozen running data.

## Data and checkpoint contracts

`tools/run/freeze_event_recipe.py` admits a separate v6 recipe binding the
complete39526-sequence v5 parent, existing v2 event annotations and original
stats. It never edits those inputs. TRAIN condition dropout15% is deterministic
by seed/step/rank and leaves base source/stratum/window draws unchanged. DEV
uses explicit event conditions and reports an additional unknown-event control
on the same samples. These are demonstration-derived proximity task annotations,
not measured force or certified intentional contact/release. Online conditions
must come from the task scheduler, not future realized rollout observations.

`tools/run/migrate_v5_checkpoint.py` is the only v5→v6 expansion path. It checks
the original data, batch/world/config; keeps all original weights, moments,
steps, elapsed budget and per-rank RNG; and initializes only event parameters.
Selection resets because conditioned DEV has a new input contract. Normal
`--resume` accepts only a matching v6 checkpoint/recipe/config/world. v5 cannot
silently load a v6 checkpoint. Migration provenance follows subsequent saves.

## Bounded four-GPU readiness check

The user requested temporarily stopping v5 and testing sustained v6 training,
then explicitly chose **restore v5 after the test**. The owned controller
`tools/run/probe_with_v5_restore.py` saves/stops the identified v5 run, migrates
its final complete checkpoint, performs8 recovery updates +1000 sustained
updates +8 recovery updates, and restores v5 from its original stop checkpoint
in a fresh output directory. v6 trial updates are never consumed by v5.

All phases use four GPUs,local128/global512,bf16,8 spawn workers/rank and the
original LR1e-4/warmup history. Check event gradients/updated weights, finite
losses, real DEV/checkpoint publication, four-rank signatures and actual v6
resume. This establishes bounded operational readiness, not80k-step reliability
or a causal event-conditioning/robot benefit.

See [the continuation probe](docs/experiments/probes/P-20261019-wam-v6-event-continuation.md)
and [current state](docs/STATE.md). For remote execution use graspenv and set
REF2DEX_V5_BASE_CODE_ROOT to the existing frozen v4 dependency tree; archive
both the v5 shared primitives and this v6 task together.
