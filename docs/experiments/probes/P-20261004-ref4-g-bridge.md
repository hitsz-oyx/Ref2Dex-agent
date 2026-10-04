# P-20261004 Ref4 G bridge

**Type.** Decision Probe.

**Question.** After a history representation `H` is available, does ground-truth
future consequence `E` and then interaction `I` provide independent information
for predicting the long-horizon value target `G`? This is the minimal bridge
test suggested by `docs/ref/ref4.md`; it does not train a policy or a
consequence predictor.

**Data and contract.** The probe reuses the corrected Gate-1 assembled dataset
`tmp/e260_all4_h16.pt`, which contains 58,111 windows from four Inspire
physical runs. The source shards are `pre_env_step`, use exact simulator
Monte-Carlo return-to-go, and have the same source namespace. The first `K=8`
future rows are used for E/I. `G` is the stored `return_to_go` target without a
bootstrap. H is the ten-step history of state, previous action, context and
progress. E and I are the stored object-frame consequence sequences.

The split is fixed by `(source_namespace, source_run, episode_id)` with seed
`20261004`: 22 held-out episode groups and 88 training episode groups (46,500
training rows; exact counts are recorded in the JSON artifact). Each arm uses the same
episode split, a GRU encoder per block (hidden 48), and a two-layer MLP value
head. Three arms are fit independently for 16 epochs on GPU 6:

```
H -> G
H + E_GT -> G
H + E_GT + I_GT -> G
```

The primary metric is held-out episode-balanced MAE. The bootstrap interval is
descriptive over held-out episode groups. No matched predicted-consequence
checkpoint was available, so predicted E/I was not evaluated.

**Results.**

| Bridge input | Episode-balanced test MAE |
| --- | ---: |
| H | 24.485 |
| H + E_GT | 21.376 |
| H + E_GT + I_GT | 22.075 |

Relative to H, H+E improves 12.7% (episode bootstrap delta CI95
`[-0.685, 10.085]`) and H+E+I improves 9.8% (CI95 `[0.074, 6.439]`). The
direct test of I's independent contribution, H+E+I versus H+E, is negative:
`-0.699` MAE reduction, or `-3.27%`, with CI95 `[-4.242, 1.330]`.

**Decision.** `PROMISING` for an E-to-G bridge direction on this split, but
`UNPROMISING` for adding the current 80D interaction sequence after E. The
result does not justify making I a core G input yet. Keep I as an auxiliary or
diagnostic target while the I surface-token route is tested separately. Do not
infer predicted-I utility, policy utility, or a formal validation claim from
this single Probe.

**Artifacts.**

- Implementation: `scripts/probe_ref4_g_bridge.py`
- Result: `tmp/P-20261004-ref4-g-bridge.json`
- One-epoch wiring smoke: `tmp/P-20261004-ref4-g-bridge-smoke2.json`

Validation performed: Python bytecode compilation, one-epoch GPU smoke, and
the fixed 16-epoch GPU probe. User documentation migration files and
`AGENTS.md`/`docs/CAMPAIGN.md` were not modified by this route.
