# WAM v6 state — 2026-10-19

The user authorized moving planned event-goal work into this task and a bounded
four-GPU sustained training check, then restoring the original v5 training.
Frozen d37e2320 implements separate recipe/migration/strict resume entrypoints.
The combined v5/v6 suite passes48 tests. Owned controllerPID3068448 is running
the four-GPU readiness check. v5 saved/stopped normally at15120; its original
checkpoint is the restoration parent. v6 expansion preserved all564 original
optimizer states and original weights, adding only4 event parameter tensors.
First real v6 update15121 completed with finite loss/gradient and nonzero event
gradient0.00787 after clipping; new output weight norm0.02206. Full sustained
training and v5 restoration remain pending; see the experiment card.

Shared representation remains768/12heads/12layers,joint21/K24,778 hand samples,
2048 object/C points,5mm PTv3 and original horizon normalization. New event
goal encoder has50304 parameters and no KNN or sequence expansion. v5 baseline
model/config/training files are untouched.

Inputs: remote v5 final recipe and `data/events-candidate-20261019-r2/` under
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/`.
v6 outputs will live under the separate remote root
`/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v6-20261019-r1/`.

Trial:8-update recovery,1000-update sustained run,8-update recovery; then
restore v5 from its own saved stop checkpoint, with no v6 updates mixed in.
The [experiment card](experiments/probes/P-20261019-wam-v6-event-continuation.md)
records the protocol, provenance, outcomes and limitations.
