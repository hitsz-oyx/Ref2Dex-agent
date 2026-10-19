# WAM v6 state — 2026-10-20

User now requests continuing v6. v5 after-v6-probe-r3 saved/stopped normally at
15723; its complete checkpoint remains available. Strict v6 continuation uses
the learned event model's own readiness checkpoint16136, frozen d37e2320,
not a new v5 migration. Current four-GPU run:
pretrain-nine-v6-events-20261020-r1, supervisor3083547/torchrun3083564.
Verified RUNNING at16520, global512, loss0.77588,1.114 updates/s, peak allocated
18.57GiB; GPUs99–100% and about23–27GiB physical memory. Remaining80k updates
estimated17–19h including scheduled DEV/saves; progress ETA itself omits some
periodic overhead. Neither full completion nor event efficacy is established.

Model/config/recipe remain unchanged:768/12heads/12layers,joint21/H4/K24/30Hz,
surface778/object2048,voxel5mm, original TRAIN-only horizon normalization,
AdamW LR1e-4, existing warmup history and optimizer/RNG/elapsed. Restore moved
568 fused AdamW step tensors to parameter devices; strict local128/global512
and world4 binding passed. Original80000/cumulative604800s bounds retained.
Event encoder50304 parameters/dropout15%, no KNN or new tokens.

## Target outlier diagnosis

Actual reader index3150491/sample[29384,0,35], Shadow s4-34534, produces8.92525m
maximum scene-point displacement. Original local rollout and remote compact
payload object/time/hand-root/reset/success arrays match exactly; compact object
translations also match the frozen cache exactly. At frame36 the object starts flying away; maximum speed14.087m/s,
h24 center displacement8.85587m, eventual y17.023m. Clock is regular30Hz, no
reset, rotation orthogonality error1.19e-7, query radius43.65mm. This is an
inherited simulator-runaway segment, not a v6 event/scale/frame-transform bug.
Actual current parent recipe lacks strict motion-admission metadata; historical
cleaning validates finite poses/clocks and observed workspace, not future
runaway motion. It admits ticks3,11,19,27,35. The last3 windows cross onset36.

Read-only audit7a817817 scanned all39526 sequences and actually admitted
TRAIN3218740/DEV545459 windows in67.28s. Object-center displacement>2m occurs
in29 TRAIN/5 DEV windows, all Shadow. A Shadow-only candidate prefix cutoff at
first known observed center speed>5m/s identifies60 TRAIN/10 DEV windows in25
sequences. Weighted TRAIN exposure is0.00275% of draws (about1.4% of global512
batches under independent draws), so rare samples can still cause visible spikes.
Candidate exclusions are prepared but NOT applied; no running recipe/cache was
modified. A clean data continuation needs a separately admitted recipe and
explicit checkpoint data-contract transition. No scratch retrain is implied by
these sparse findings. Large net displacement alone is not a bad human label.
HOT3D s6-00372 has2 additional TRAIN windows crossing known observed center
transitions12.10/12.54m/s; retained as tracking candidates, not classified as a
confirmed simulator defect. The scan covers center translation, not full hand,
angular or surface-point motion quality.

## Prior readiness evidence

The8+1000+8 probe completed15120→16136, checkpoints/recovery/signatures passed.
CPU audit verified564 original optimizer states at16136,4 event states at1016,
4 RNG payloads and exactly unchanged horizon-normalizer buffers. Latest probe
DEV handEPE33.167mm versus unknown-event33.285mm: a small same-model diagnostic,
not a matched-training causal gain. Combined v5/v6 suite48 tests passed;
2 new motion-audit tests cover future-only runaway, rigid frame invariance and
observed-frame clock/masks. No model/training code changed for this switch.

Remote root:/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/.
Current run:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/.
Switch:outputs/switch-to-v6-20261020-r1/report.json.
Motion scan:outputs/target-motion-audit-20261020-r1/summary.json.
Forensics and candidate exclusions:outputs/target-motion-forensics-20261020-r1/.
Small report/plot copies:outputs/ptv3-shared-wam-v6/pretrain-nine-v6-events-20261020-r1/ in local repository.

[Current training and diagnosis](experiments/probes/P-20261020-wam-v6-mixed-pretrain.md)
· [completed readiness check](experiments/probes/P-20261019-wam-v6-event-continuation.md).
