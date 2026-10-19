# WAM v6 state — 2026-10-19

The user authorized moving planned event-goal work into this task and a bounded
four-GPU sustained training check, then restoring the original v5 training.
v6 implementation and new recipe/migration/strict resume entrypoints are ready
for local tests and remote verification. Production v5 continues while code is
prepared; no version6 update has yet run.

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
