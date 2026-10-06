# cm-pointflow-effect-pretrain

Current user-directed route on branch `cm-pointflow-effect-pretrain`: OakInk2
30Hz /24-step action-conditioned multi-object world model. Program anchor plus
0.5m current geometry-center local objects; fixed right/left11semantic points
and masks;512canonical object surface points. Predict every local object's
24-step SE(3) and analytical point trajectory. No RGB, KNN, Policy/Evaluator/Y.

[User architecture](docs/user/架构.md), [frozen implementation contract](docs/WM30_DESIGN.md),
[completed code and real-data interface checks](docs/WM30_INTERFACE.md),
[matched full-corpus experiment](docs/experiments/probes/P-20261007-oakink2-wm30-k24.md).

Data acquisition and full627 preparation are running in
`outputs/cm-pointflow-effect-pretrain/oakink2-wm30-k24-20261006/`.
Three independent H/H+A/H+shuffle(A) arms use the same initialization and
40000updates, GPU0/1/2, effective batch16 and shared24h cap. H+A additionally
receives validation/test shuffle. The launcher waits for all verified data,
processed split integrity and free GPUs. Runtime stage/PIDs/deadline are in
`group_status.json`; each arm writes `progress.json`, manifests and checkpoints.
Full training has not started while the corpus gate is pending.

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

Only data acquisition and engineering checks have run. No pretraining or PPO
training has started. Root Mission/Campaign remain authoritative; future model
training uses available GPUs within the global four-GPU boundary.
