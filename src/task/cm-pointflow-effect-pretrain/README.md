# cm-pointflow-effect-pretrain

Current user-directed route on branch `cm-pointflow-effect-pretrain`: OakInk2
observed human hand trajectory -> object effect. Keep single and both hands in
fixed right/left slots, 11 semantic points per hand with validity masks, and
512 canonical object surface points. No RGB, dataset mixing or PPO training.
Observed future hands provide predictive supervision; they do not establish
causal intervention or policy utility.

The annotation-only 100-sequence engineering audit is documented in
[frozen OakInk2 protocol](docs/OAKINK2_DATA_PREFLIGHT.md) and
[completed results and next decision](docs/OAKINK2_DATA_RESULTS.md).
Runtime data and caches use the declared artifact root
`artifacts/cm-pointflow-effect-pretrain/oakink2-data-preflight-20261006/`.
Audit code is in `tools/audit/audit_oakink2_sequences.py`; cached continuity and
filter overlap checks are in `tools/audit/summarize_oakink2_continuity.py`.
Download code is in `tools/run/download_oakink2_audit.py`. Its run directory
requires pinned Hugging Face `dataset_info.json`, `anno_tree.json`, and
`root_tree.json` metadata before acquisition. Run identities and checksums are
saved in manifests; reruns use fresh output/manifest directories.

OakInk2 mocap is120Hz: eight consecutive frames cover66.7ms. Sequence-level
holdout and interaction/static balancing must precede predictor training.
Root Mission/Campaign remain authoritative. No large pretraining has started.

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
