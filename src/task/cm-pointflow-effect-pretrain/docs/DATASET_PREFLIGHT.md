# SPIDER Inspire data preflight — 2026-10-06

## Decision and execution scope

Question: does the released SPIDER data expose physical hand/object states,
executed robot controls and portable geometry sufficient to build point-flow /
object-effect supervision? Cheapest check: one lexicographically first Inspire
trajectory from each of four sources, exact XML dependency closure, raw checksum
validation and saved-state FK. This is a Blocker engineering check before large
training, not a policy-utility Probe or scientific claim. User explicitly requested
branch `cm-pointflow-effect-pretrain` and data acquisition following their ref1.
No additional branch or large training was initiated.

Decision: **sample-level offline geometry/flow readiness PASS**; **existing Isaac
Gym control replay remains unverified and requires adaptation**. Keep the raw data
and reusable acquisition/audit tools. Large training requires a separate frozen
training contract and corpus acceptance, not automatic extrapolation from four
samples. Root Cm causal-utility goal is unchanged.

Budget: sample download <=2GiB selected files, cumulative transfer attempts
<=900s; isolated audit environment and reports within global300GB. CPU-only
because this checks files,750 saved-state FK frames and tiny geometric arrays,
not neural model computation or trajectory optimization. Zero GPU training.

## Pinned inputs and local outputs

- Dataset: [retarget/retarget_full](https://huggingface.co/datasets/retarget/retarget_full),
  revision `1bb88b0c32b33eb868def58333a516ad2f2aba30`.
- Dataset checksum inventory:89902 files. Inspire robot trajectories:1946,
  with source counts274/1363/234/75. The other robot families are excluded.
- Source: [SPIDER](https://github.com/facebookresearch/spider),
  revision `44717007de41cbef7565dff7ff9f4453557a2d3d`; pinned pyproject and official
  loader retained in `official_source/`. Loader/source manifest includes hashes.
- Successful acquisition commit `03e23cc`; audit commit `a6c6c96`.
- Run root: `outputs/cm-pointflow-effect-pretrain/spider-data-preflight-20261006/`.
- `download_samples-r4.json`:120 files,31647762 bytes, each SHA256 matches the
  pinned official checksum inventory; raw layout remains portable.
- `sample_audit.json`:model loading, dimensions, time grid, finite arrays,
  unit object quaternions and fixed-identity mesh point flow.
- `pointflow_smoke/{source}.npz`:104 hand vertices +8 object vertices/frame,
  qpos and raw controls, h8 displacement with fixed point identity. Vertex samples
  demonstrate wiring; they are not uniform surface samples or a final dataset.
- `environment_freeze.txt`, `audit_manifest.json`, `resource_summary.json`
  record runtime packages, code/output hashes and measured resources.

## Observed data contract

| Source | Full Inspire count | Sample frames | qpos/qvel/ctrl widths | Object rise (m) |
| --- | ---: | ---: | --- | ---: |
| DexYCB lifted | 274 | 175 | 25 /24 /18 | .33208 |
| HOT3D v2 | 1363 | 250 | 25 /24 /18 | .24175 |
| HRDexDB 24f | 234 | 175 | 25 /24 /18 | .28186 |
| OakInk lifted | 75 | 150 | 25 /24 /18 | .24771 |

All four scenes load unchanged with MuJoCo3.7.0. Arrays are time-blocked
(e.g.7x25x25 qpos); flatten the first two time dimensions only. The saved control
stream is100Hz, while optimizer `ctrl_dt=.25` and reference `ref_dt=.08` have
different meanings. Time is monotonic with .01s spacing. qpos ends in object
XYZ (metres) + quaternion wxyz. The hand has6 wrist+12 finger coordinates.
750 total frames /718 valid h8 displacement pairs, no reset stitching.
The smoke's h8 means .08s on this100Hz clock, NOT previous Isaac30Hz h8=.267s.
No future control chunks or training labels have been asserted yet: establish
pre-step/post-step control timestamp semantics before action-conditioned fitting.

## Problems found and fixed or deferred

1. Python urllib3.8 fails some asset CDN redirects with SSL EOF despite three
   retries. Curl successfully fetched the identical apple mesh and its checksum,
   then all remaining dependencies. Final transfer11.82s; failed manifests and
   logs preserved, validated files reused. This is a transport issue, not corrupted
   dataset bytes. Total manifest download wall221.79s across four attempts.
2. Current SPIDER source requires Python>=3.12 and pins MuJoCo3.7.0; existing
   Isaac Gym environment is Python3.8/MuJoCo3.2.3. An isolated local `.venv`
   using available Python3.12 installs only MuJoCo3.7.0/PyYAML and dependencies.
   No `sudo`, system package installation, existing environment change or full
   `uv sync` of the optimizer stack is needed for this data check.
3. Point-flow audit initially selected only group1 visual geometry, but object
   visual mesh is group0. Explicitly select `right_object_visual`; final forward
   FK and h8 flow are finite for all four sources. Failed audit log retained.
4. MuJoCo ctrl is actuator control, not the native normalized PPO action. Joint
   ordering must be mapped by names; quaternion order, root frame and physical
   sampling rate must be mapped explicitly. MuJoCo and native PD gains differ.
5. Native Inspire enforces finger couplings at reset and action conversion.
   Actual saved SPIDER joint poses violate those same couplings; maximum sampled
   discrepancy reaches1.01693rad. This is a material representational/dynamics
   mismatch, not merely18-dimensional action renormalization. Direct raw control
   replay through existing PPO action conversion cannot establish equivalence.
6. No recorded actual contact forces are exposed in the NPZ. Geometric contact
   or recomputed MuJoCo dynamics are different labels and require a defined
   contract; no native Isaac contact equivalence was tested.
7. The release is success-selected and some lift references are synthetic;
   its existing scene validation establishes saved-state portability, not new
   control replay. Source asset/collision equivalence and source provenance
   limitations remain. These data alone do not cover failed grasp recovery or
   establish grasp success rates.

## Next decision

Expand the accepted Inspire corpus with checksum/dependency/shape checks and
freeze the pretraining inputs, effect targets, physical horizon and episode /
source splits. Keep all windows from each original source episode in the same
split (the full release shares source episodes across robot retargets).
For Isaac integration, first map joints/root/control and decide how to handle
native coupling without overwriting existing environment behavior. A bounded
control replay must measure state/object drift before using these records as
native-transition labels. Offline MuJoCo physical flow can already be extracted
on these samples, but policy benefit and transfer remain OPEN. No large run,
new resource allowance or core claim change is implied by this engineering PASS.
