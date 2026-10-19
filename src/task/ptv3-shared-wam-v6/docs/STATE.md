# WAM v6 state — 2026-10-20

The separate event-goal implementation is ready for a separately chosen long
run. Frozen d37e2320 passed the requested four-GPU bounded continuation:
8 recovery +1000 sustained +8 recovery updates,15120→16136. All phases saved
complete checkpoints and had equal four-rank parameter signatures. CPU audit
368feea9 verified564 original optimizer states at16136,4 event states at1016,
all4 RNG payloads, strict v6 resume binding and exactly unchanged original
24-step horizon-normalization buffers. The combined v5/v6 suite passes48 tests.

Steady throughput was about1.115 updates/s, close to v5's1.118; sustained
GPU forward+loss258.10ms, backward596.53ms, optimizer29.57ms, peak allocated
18.81GiB. Sustained loop including startup, DEV and saves took1052.25s.
Latest fixed DEV at16129: objective1.28647, handEPE33.167mm, trajectory
selection score17.008mm. At15129 these were1.29009,33.638mm,17.251mm.
Same-model unknown-event DEV handEPE was33.285mm at16129: only0.118mm worse.
There is no matched training control and no event-efficacy or robot claim.

A rank0 local last-batch loss26.098 at16128 is retained in the record.
Exact TRAIN-draw target inspection found source4 sample[29384,0,35] with
8.925m maximum future object displacement and normalized scene-target
mean-square2974.71; full batch scene-target mean-square24.011. This is a
strong target-outlier explanation, not a model-forward loss attribution.
No nonfinite failure occurred; next resumed global loss was0.7454 and final
local loss0.7530. Trace that sequence before making a future data-admission
change; neither frozen data nor running v5 was silently filtered.

Per the user's explicit choice, v6 is stopped and its checkpoints retained.
v5 restored from its own15120 checkpoint on original frozen ac951370 code,
with no v6 updates mixed in. It was verified RUNNING at15260, global512,
about1.116 updates/s; all4 GPUs were100% utilized (about21–27GiB each).
Active baseline run: pretrain-nine-v5-width768-after-v6-probe-20261019-r3.

Shared representation remains768/12heads/12layers,joint21/K24,778 hand samples,
2048 object/C points,5mm PTv3 and original horizon normalization. New event
goal encoder has50304 parameters and no KNN or sequence expansion. v5 baseline
model/config/training files are untouched.

Inputs: remote v5 final recipe and data/events-candidate-20261019-r2/ under
/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/.
v6 root: /mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/.
Final checkpoint: outputs/ptv3-shared-wam-v6/event-continuation-readiness-20261019-r1/resume-recovery/step016136.pt.
Objective-best: sustained/step016000.pt; trajectory-best: resume-recovery/step016129.pt.

The [experiment card](experiments/probes/P-20261019-wam-v6-event-continuation.md)
records the protocol, provenance, outcomes and limitations.
