# goal-point-wam

This Task implements Goal-Point-WAM on the frozen OakInk2/GRAB/ARCTIC
main-three corpus, with a separate EgoDex video point-supervision route.

The scene encoder consumes only current/history scene points. The trajectory
branch supports the historical hand endpoint proxy and the `object_points`
contract requested in ref1_1: 32 paired current/t+24 target-object points,
expressed in the single current anchor frame. The target flag selects the
object; future hand endpoints are absent from this goal contract. The world branch receives the executed future
hand trajectory and predicts the existing per-object rigid effect; it never
receives goal tokens. RGB, video, robot H328, PD, policy and evaluator losses
are outside this phase.

Use the existing mixed manifest and train-only normalization statistics:

After retiring the former worktree on 2026-10-11, use the path-relocated
`data-main-worktree` view below. Index/source hashes and normalization identity
are unchanged; the original `data` manifest retains its historical bytes.

```bash
PYTHONPATH=src/task/goal-point-wam/src:src/task/cm-pointflow-effect-pretrain/src \
python3 src/task/goal-point-wam/tools/run/train_three_domain.py \
  --data outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/data-main-worktree \
  --stats outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/norm_stats.json \
  --init-pw outputs/cm-pointflow-effect-pretrain/pointworld-main3-sharedstats-20261009/training-r1/train-action/final.pt \
  --output outputs/goal-point-wam/<run_id> \
  --require-cuda
```

`--smoke --steps 4` runs the engineering-only contract, gradient, checkpoint
and goal-isolation check. Probe outputs belong under
`outputs/goal-point-wam/<run_id>/` and must not overwrite the historical
PointWorld outputs.

Fresh runs require `--init-pw`, or explicit `--random-init` for a control.
`--resume` is separate and verifies WAM code/data/config/stats identities.
The repaired 50k-step PW final checkpoint initializes all PTv3 backbone and
18-to-128 scene projection tensors. WAM goal/action/world/trajectory heads and
its differently structured object projection remain fresh. Temporal PW weights
are reused in the scene-only spatial encoder; this is weight transfer, not a
reproduction of PW's action-fused temporal forward. World memory now retains
all 528 executed-action tokens and their validity masks.


Object-goal runs use `--config configs/three_domain_object_goal_probe.json`
(relative to this Task) with the same trainer. The cheaper train-only contract
Probe is `tools/audit/probe_object_goal.py`; it reuses the saved six-window
manifest, independently checks object endpoints against canonical points/raw
poses, performs 300 fixed-batch updates from PW initialization, and reports
symmetric object-goal translations with identical cached scene/action memory.
The response gate is an exploratory mechanism screen, not a physical guarantee
that hand movement should equal object movement. No robot/RL utility is tested.


Goal geometry diagnostics use `tools/audit/probe_controlled_goal.py`. Its
analytic translation labels only test conditional learning; checkpoints marked
`analytic-diagnosis.v1` must never initialize real-data training. Optional
`goal_coordinate_scale_m` defaults to1 (historical meter coordinates);
`object_goal_encoding` defaults to `endpoint`, with exploratory `start_delta`
supported for paired point geometry. Both keep historical parameter keys.

The real-data units comparison uses `three_domain_object_goal_{raw,scaled}_600.json`.
`tools/audit/evaluate_object_goal.py` loads its final real-data checkpoint and
saved validation indices, checks data/stats/model identity, and compares true
goal versus static object goal with identical cached scenes. It also reports
zero hand displacement and object translation/rigid-follow geometric references;
these references are prediction baselines, not physical controllers.


`tools/audit/probe_hand_history.py` checks semantic observed-hand dynamics
against raw histories and reports zero/velocity forecast references on the saved
validation panel. `tools/audit/probe_compact_forecast.py` then trains matched
H-only/H+object-goal23.9k-parameter diagnostic MLPs on the original1200-sample,
600-update stream. Current-hand history is exposed as `start_hand_history`
(velocity, acceleration, three-step displacement); existing WAM configs ignore
this additional observed-only field. Compact checkpoints are diagnostics, not
WAM/policy initialization. History prediction and goal utility are separate gates.


Matched object-goal injection uses `goal_injection: memory|query|none` and a
common `goal_query_adapter: true`. Memory uses encoded goal tokens; query masks
those tokens and adds a pooled start/end embedding to trajectory queries; none
masks both routes, including the encoded null token. Historical configs without
the adapter keep their existing checkpoint keys. `observed_hand_query: true`
adds a per-keypoint adapter from current XYZ and observed9D history only, with
missing-hand masks. Its allocation preserves the original core initialization
and RNG. It receives neither goal nor future hand/action labels.

The completed batch64/1200-update configs are `three_domain_goal_injection_*_large.json`
and `three_domain_hand_goal_*_world{0,1}.json` under `configs/`. See the
[goal route comparison](docs/experiments/probes/P-20261012-goal-point-wam-goal-injection.md)
and [hand/goal/world-loss comparison](docs/experiments/probes/P-20261012-goal-point-wam-hand-goal-interaction.md)
for exact config filenames and component-specific results. These remain local
single-seed Probes; the useful candidate is observed hand + memory goal + world1.

`tools/audit/compare_goal_injection.py` and `compare_hand_goal_interaction.py`
bind actual final checkpoint hashes, indices, initialization, budgets and frozen
EPE before comparison. `audit_world_isolation.py --code-root <matching-code-root>`
is a no-training engineering diagnostic: it separates full-forward same-input
noise from world decoding with identical cached scene/action memory and checks
absence of a goal gradient. Original full-forward threshold failures remain
recorded; cached independence does not establish physical world accuracy.


Single-model scale-up uses `tools/run/launch_distributed.py` with
`configs/three_domain_hand_memory_ddp_large.json`, four idle GPUs and a frozen
panel generated by `tools/audit/freeze_validation_panel.py`. It launches NCCL
DDP through `train_distributed.py`, with128 samples/rank/global512, independent
sample occurrences per rank and synchronized gradients. All4 ranks evaluate;
metrics aggregate actual valid-point/object counts. Rolling checkpoint state
includes per-rank RNG; final bare model keys remain compatible with the existing
object-goal audit. The launcher records ownership and stops its own ranks on
input drift, foreign GPU conflict, deadline or storage threshold. See the
[scale-up card](docs/experiments/probes/P-20261012-goal-point-wam-ddp-scale.md)
for the actual run and resources. This single-seed run has no matched control.

The current continuation uses GPUs0/1/2/3, restoring step400 through `--resume`
and keeping the original supervisor `--deadline`; see the card for exact paths.


Native EgoDex auxiliary training uses `tools/audit/prepare_egodex_actor.py`
with completed official-TRAIN acquisition manifests. It admits native H4/K24
confidence-masked hand labels, fixes the current camera frame and splits by
session. `tools/run/launch_egodex_auxiliary.py` supervises
`train_egodex_auxiliary.py` on one or four GPUs from `--init-wam`: three full
main-corpus updates followed by one native actor-only update. EgoDex has null
object goals and bypasses action/world forward; object-goal/world supervision
continues on the original main-three corpus. Local and remote configurations
are `configs/egodex_native_auxiliary_{local,remote}.json`.
See the [admission and execution card](docs/experiments/probes/P-20261012-goal-point-wam-egodex-auxiliary.md)
for qualified inputs, limits and runtime evidence.

Direct video point supervision uses `src/goal_point_wam/point_supervision.py`
and `tools/run/train_video_points.py`: the same actual PTv3 parameters serve
native rigid and video point batches. Native SE(3) predictions are decoded to
points; video tracks use a direct point head with no pose or mesh requirement.
Both use the common masked metric point loss. World conditioning remains
executed future hand actions; actor conditioning remains paired goal points.
Weight transfer from a real WAM checkpoint is explicit, with a new point-model
checkpoint kind and fresh optimizer. Old checkpoints are not silently resumed.

`tools/audit/prepare_video_points.py` reuses existing offline teachers only for
labels, re-runs observed-prefix depth for current inputs, and keeps original
native2x11 hand semantics. Its output is causal but engineering-only until
depth/track quality and held-session admission are reviewed. Optional current
target mask supplies paired goals; otherwise videos supervise point-world with
null goals. Pass `--smoke` to the trainer for four finite-gradient mixed updates
without saving a checkpoint. Scientific training rejects engineering manifests.
Single GPU or `torchrun --nproc_per_node=2` use the same trainer; no copied
backbone per data source. See the
[video point card](docs/experiments/probes/P-20261012-goal-point-wam-video-points.md).

`tools/audit/evaluate_horizons.py` reports each of the24 hand prediction horizons
and nearest1--8/middle9--16/far17--24 aggregates on the frozen validation panel.
The final10000 audit was completed before the optional dense-feature interface
changed the model source hash. Re-running old strict identity audits requires
their original committed model source, although historical parameter keys are
unchanged.


Video point supervision uses `tools/audit/prepare_video_points.py`,
`tools/audit/audit_video_tracks.py`, and `tools/run/train_video_points.py`.
The video and native branches share the same PTv3 parameters; video uses
a direct point trajectory head, while native rigid effects may retain their
SE(3) head decoded to the common masked point loss. Existing WAM checkpoints
initialize this new model through explicit weight transfer and a fresh
optimizer, rather than resuming the native checkpoint.

For offline pretraining, `--label-depth offline-teacher --window-offset 0|3|6`
extracts input and future labels from one previously cached clip reconstruction.
It preserves the native camera/hand clock and records reconstruction context
and calibration provenance. This mode may use future clip context for depth
and is explicitly marked `offline_pretraining_only`; it does not certify an
online causal perception pipeline. The optional `rolling-prefix` mode has
different causal context and is not the current training route.

An engineering corpus remains `training_allowed=false`. Scientific mixed
Probes require a separately admitted manifest, frozen train/dev recording
sessions, geometry/calibration screens and the frozen native validation panel.
See the [video experiment card](docs/experiments/probes/P-20261012-goal-point-wam-video-points.md)
for actual admission counts, code identity, run outcomes and limitations.

`--trajectory-depth /path/to/depth.npz` optionally reuses one once-calibrated
offline geometry span across several cached tracking clips. The sibling
`result.json` must declare a completed engineering geometry, original source
video, frame IDs, calibration and depth hash. Extraction validates the raw
camera/intrinsic/clock and reads existing depths without a model run or scale
refit; `audit_video_tracks.py` checks the same shared geometry and provenance.
This route remains engineering-only until separate quality admission. A cached
span may cover only part of a recording; see the video card for actual coverage.
