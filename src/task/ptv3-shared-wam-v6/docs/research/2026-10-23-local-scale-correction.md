# 2026-10-23 local-data scale correction

This note corrects the earlier suggestion that a local full-data retraining run
was immediately available. The local WAM-v6 recipe is a fixed engineering
subset; it is not the nine-source full recipe.

## Facts

- The prepared local cache contains **1,177 sequences**: GRAB 1,132 and 45
  sequences from the other eight sources. It contains **33,672 TRAIN** and
  **3,356 DEV** windows. The preparation and subset manifest explicitly keep
  **38,349 sequences unprepared**.
- The later local-expanded recipe reuses those same 1,177 caches. It expands
  GRAB TRAIN windows to **207,579** (from 33,672), keeps DEV at 3,356, and
  does not create new sequence caches. This is a sampling/window-scale change,
  not full-data preparation.
- The current flow-matching `process.json` records `target_updates=30000` and
  `max_cumulative_training_seconds=43200`. The final training log reached
  step 11,380; the newest durable checkpoint is `step011000.pt`.
  `expected_final_step=12205` is the time-limit forecast in that receipt, not
  a persisted checkpoint. The run is stopped and must not be resumed under the
  old goal contract.
- At this check, `/home2` has **141 GiB free** (`22T` filesystem, `21T`
  used). Existing large output trees include about 141 GiB for
  `cm-pointflow-effect-pretrain`, 132 GiB for `ptv3-shared-wam-v2`, and 65 GiB
  for `ptv3-shared-wam-v6`.

## Storage estimate for expansion

The prepared 1,177-sequence cache occupies 6,031,884,656 bytes, or about
**4.89 MiB per sequence**. A linear estimate for a target of `N` prepared
sequences is therefore `4.89 MiB × N`; the incremental column is in addition
to the existing 1,177-sequence cache.

| prepared sequences | estimated total cache | additional cache from current subset |
|---:|---:|---:|
| 1,177 | 5.6 GiB | 0 GiB |
| 2,000 | 9.5 GiB | 3.9 GiB |
| 5,000 | 23.9 GiB | 18.2 GiB |
| 10,000 | 47.7 GiB | 42.1 GiB |
| 20,000 | 95.5 GiB | 89.8 GiB |
| 30,000 | 143.2 GiB | 137.6 GiB |
| 38,349 (registered full recipe) | 183.0 GiB | 177.4 GiB |

These are planning estimates, not a reservation: source mix, temporary
conversion peak, manifests, and checkpoints can increase the requirement. At
141 GiB free, the registered 38,349-sequence preparation does not fit; keeping
the campaign's 20 GiB safety floor leaves only about 121 GiB for new writes.
The full estimate would exceed that by roughly **56 GiB**, before temporary
peak overhead. No archive or deletion is authorized by this note.

## Decision gate

There are two bounded scale choices after the goal contract is repaired:

1. Continue the same 1,177-sequence subset for 24–48 hours, starting from a
   preserved checkpoint, with a fresh time budget and an explicit receipt.
2. Expand to a declared `N` only after the owner chooses what historical
   outputs to archive or delete and a fresh free-space check passes. The owner,
   not this task, must choose that reclamation.

Until the `object_goal_valid` hard-code and the train/inference goal-semantic
alignment are repaired and verified, both choices remain **HOLD**. No training
or new cache preparation is started by this correction.
