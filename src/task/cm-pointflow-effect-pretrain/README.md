# cm-pointflow-effect-pretrain

Main training completed (2026-10-08): user ref3 uses OakInk2/GRAB/ARCTIC main
supervision only, on GPUs1/2 with per-rank batch64/global128. The stopped
four-source latest14250 initializes model weights; AdamW/schedule/draw reset.
All50000new updates completed before2026-10-08 10:00 Asia/Shanghai;
latest/final50000 and best46000 are preserved. Fixed-panel source macro
moving-anchor h24 EPE34.480→27.126mm (best26.786mm). Both ranks agree;
workers and recovered monitor exited. ContactPose is excluded from new
main sampling/loss, but its earlier learning history remains in the imported
parent. This is not a clean ContactPose-free causal ablation. Separate
ContactPose auxiliary now has a separate [matched Probe](docs/experiments/probes/P-20261008-contactpose-transport-auxiliary.md)
and [bounded runner](tools/run/train_contactpose_transport_auxiliary.py).
Both arms share main/transport batches; only the history-only transport loss
coefficient differs. Future ContactPose hand chunks are absent from auxiliary
forward inputs; this tests incremental rigid-transport utility from the same
parent. Weak-data training remains deferred. EPIC
remains candidate_only after coordinate/validity/dual-hand/gap audits.
See the [current mixed pretraining card](docs/experiments/probes/P-20261007-pointworld-multisource.md)
for per-source moving-anchor metrics, exact artifacts and frozen identities.

Current development branch: `cm-pointflow-effect-pretrain` (renamed from
`agent/pointworld-ref8-fix` on 2026-10-10). Local worktree:
`/home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent-wt-cm-pointflow-effect-pretrain`.
The former `Ref2Dex-agent-wt-ref8` path is retained as a compatibility symlink.
See the [completed ref8 run and rename note](docs/experiments/probes/P-20261009-pointworld-main3-sharedstats.md).

Architecture:
PointWorld-small unified spatial encoder for OakInk2 30Hz /24-step
action-conditioned multi-object world model. Program anchor plus
0.5m current geometry-center local objects; fixed right/left11semantic points
and masks;512canonical object surface points. Predict every local object's
24-step SE(3) and analytical point trajectory. No RGB, KNN, Policy/Evaluator/Y.

[User ref3](docs/user/ref/ref3.md), [PointWorld design](docs/POINTWORLD_WM24_DESIGN.md),
[implementation and GPU checks](docs/POINTWORLD_INTERFACE.md),
[new matched experiment](docs/experiments/probes/P-20261007-pointworld-small-wm24.md).

External HOCap frame-index [Probe](docs/experiments/probes/P-20261008-hocap-frame-generalization.md)
completed192windows across64sequences/9subjects with fixed latest50000:
moving h24 point EPE17.805mm vs persistence63.704mm; sampled near-static
anchors22.160mm vs0.356mm. Keep these outcomes separate. Source FPS is still
unverified; this is no-fit nominal-clock inference, not a formal0.8second
external test. HOCap remains outside training/statistics/checkpoint selection.
On the exact same frozen panels, the earlier Oak-only latest10000 gives
moving h24 EPE22.421mm versus mixed17.805mm (20.59% reduction), while
near-static worsens8.039→22.160mm. This historical endpoint comparison
confounds mixed data with64250additional updates and batch/loss changes;
it does not isolate data-mixture benefit. Both fixed endpoints are preserved.

The upstream encoder source is pinned as the `third_party/PointWorld` Git
submodule at `05484826dfef74cbe278a3974179a5a16705d35d`. After cloning this
repository, run `git submodule update --init third_party/PointWorld` to obtain
the required source. Its nested DINOv3 dependency is not used by this route.

Full627 acquisition and preparation completed in
`outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.
Old WM30 workers saved and stopped on user request: H40000/H+A33464/shuffle34823,
so no matched final comparison is claimed. Original checkpoints remain.
New ref3 outputs use `outputs/cm-pointflow-effect-pretrain/pointworld-small-wm24-20261007/`.
Three independent H/H+A/H+shuffle(A) arms use identical initialization,
40000updates,effective batch16 and shared24h cap. H+A additionally receives
validation/test shuffle. Runtime stage/PIDs/deadline are in `group_status.json`;
each arm writes `progress.json`,source/stat manifests and checkpoints.

Earlier annotation-only100sequence readiness audit:
[frozen protocol](docs/OAKINK2_DATA_PREFLIGHT.md) and
[results](docs/OAKINK2_DATA_RESULTS.md). Artifacts remain in
`artifacts/cm-pointflow-effect-pretrain/oakink2-data-preflight-20261006/`.
That audit used120Hz h8; the new architecture deliberately switches to30Hz
K24=.8s. Real-data engineering checks use12existing sequences and are separate
from full training. Observed future hands provide predictive supervision, not
causal intervention or demonstrated robot policy utility.

## Earlier SPIDER feasibility work

Initial data preflight is complete: four sources / four Inspire trajectories,
120 checksummed files,750 saved states. All four portable MuJoCo scenes load;
qpos/qvel/control/time and fixed-identity mesh-vertex flow are finite. This is
sample-level engineering readiness, not full-corpus acceptance or Isaac Gym
control-replay validation. Current native joint coupling differs materially.
[Results, limitations and next gate](docs/DATASET_PREFLIGHT.md).

Historical data and isolated audit environment were stored in
`outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006/`, never Git.
Those runtime files became unavailable after the user deleted the shared
`Ref2Dex-agent-baseline` directory. Retained results are historical summaries;
replay needs reacquisition. Raw dataset revision and selected file checksums
were pinned. No system
packages or existing Isaac Gym Python environment were modified.

Acquisition (default one sample/source; `--per-source 0` requests all1946 Inspire
trajectories, subject to explicit byte/time caps):

```bash
python3 src/task/cm-pointflow-effect-pretrain/tools/run/download_spider_inspire.py \
  --run-dir outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006
```

The run directory must contain `dataset_info.json` and `checksums.json` from the
same pinned Hugging Face revision. Completed raw files are verified on resume;
failed attempt manifests remain separate. Meshes referenced by portable XML
are included; videos and the full bulk archive are excluded.

Audit using the isolated Python3.12/MuJoCo3.7 environment:

```bash
outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006/.venv/bin/python \
  src/task/cm-pointflow-effect-pretrain/tools/audit/audit_spider_samples.py \
  --run-dir outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006
```

Earlier SPIDER work was acquisition/engineering only. OakInk2 now has its own
training route above. Root Mission/Campaign remain authoritative; training uses
available GPUs within the global four-GPU boundary. No PPO is started.

## Video and tactile data feasibility (2026-10-10)

The EPIC dataset branch is merged locally, with original experiment outputs
preserved in this worktree. Keep converted EPIC packs `CANDIDATE_ONLY`: review
found missing semantic-joint validity/quality filtering, conflicting source
splits, clock and persistent-track questions, and a per-point schema that is
not supported by the existing rigid-object decoder. The reported P03_03
complete window used side-level validity and is not a qualified training
window. OakInk2/GRAB/ARCTIC remain the registered main sources.

EgoTouch is a candidate for separate tactile representation/auxiliary-target
work, with pressure/bend mapping and release completeness audited first. It
does not supply paired object SE(3) supervision or a calibrated substitute
for Isaac net contact forces. See the [source and implementation review](docs/research/2026-10-10-video-tactile-primary-sources.md).
Remote connection details are in [the Windows bridge guide](../../../docs/user/连接远程服务器.md);
current work continues on RLG. No new training or full dataset download has
been launched by this review.

The follow-up [EPIC readiness probe](docs/experiments/probes/P-20261010-video-data-readiness.md)
repaired semantic-joint/quality masks, source splits, metadata clocks, background
occlusion and permanent lost-track masking. The34 local clips have20 clock
metadata files;95 candidate starts across four temporal overlaps yield zero
complete hand windows. Keep these packs outside the native training manifest.
This is a local coverage blocker, not evidence against video learning.

The [original tactile label sample](docs/experiments/probes/P-20261010-egotouch-label-schema.md)
acquired two TRAIN bundles (485754 bytes,10 verified files). Original frame IDs
agree, but only217/441 grid cells are measured, per-record pressure/bend maxima
differ, confidence and camera calibration are absent, and both cheapest
episodes are annotated no-contact. Next qualification should use a
contact-positive episode and original clocks; neither sample is registered
for point-dynamics training.
