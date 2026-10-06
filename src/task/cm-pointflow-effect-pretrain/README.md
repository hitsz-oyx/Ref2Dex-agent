# cm-pointflow-effect-pretrain

User-directed route on branch `cm-pointflow-effect-pretrain`: investigate SPIDER
Inspire physical trajectories as external supervision for hand point flow and
object effects, then assess transfer into Ref2Dex policy learning. Offline
predictive accuracy alone does not establish Cm policy utility.

Initial data preflight is complete: four sources / four Inspire trajectories,
120 checksummed files,750 saved states. All four portable MuJoCo scenes load;
qpos/qvel/control/time and fixed-identity mesh-vertex flow are finite. This is
sample-level engineering readiness, not full-corpus acceptance or Isaac Gym
control-replay validation. Current native joint coupling differs materially.
[Results, limitations and next gate](docs/DATASET_PREFLIGHT.md).

Data and isolated audit environment live in
`outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006/`, never Git.
Raw dataset revision and every selected file checksum are pinned. No system
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
