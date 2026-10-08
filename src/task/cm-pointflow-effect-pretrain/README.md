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

Current development branch: `main` (integrated from `consequence-evaluator`). Architecture:
PointWorld-small unified spatial encoder for OakInk2 30Hz /24-step
action-conditioned multi-object world model. Program anchor plus
0.5m current geometry-center local objects; fixed right/left11semantic points
and masks;512canonical object surface points. Predict every local object's
24-step SE(3) and analytical point trajectory. No RGB, KNN, Policy/Evaluator/Y.

[User ref3](docs/user/ref/ref3.md), [PointWorld design](docs/POINTWORLD_WM24_DESIGN.md),
[implementation and GPU checks](docs/POINTWORLD_INTERFACE.md),
[new matched experiment](docs/experiments/probes/P-20261007-pointworld-small-wm24.md).

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
