# CHORD native/simulator contact geometry audit (2026-10-11)

## Decision

A full Isaac/MuJoCo contact smoke cannot run in the current worktree because `python3` has neither `isaacgym` nor `mujoco` installed, and the SPIDER sample output corpus is not present here. A **static CHORD support-function smoke does run** using the locally readable HRDexDB Inspire geometry. This is sufficient to validate the wrench math and data shape, but it does not validate simulator contact forces or native policy reward wiring.

No external data was downloaded and no external repository was modified.

## Locally available geometry and contracts

### HRDexDB repository

Read-only checkout: `/home2/wyy/oyx_ws/HRDexDB`, Git `2cddac708d7a4da1ed6a571093f61c8dadb095d4`.

The useful episode is `/home2/wyy/oyx_ws/HRDexDB/v0/inspire_f1/apple/0`:

- `object_6d/pose_*.txt`: 543 object poses, one per HRDexDB frame.
- `raw/arm/position.npy`: `(1856, 6)` arm/wrist trajectory (available).
- `raw/hand/right_commands.npy`: `(1056, 6)` object-array hand commands (available).
- `raw/hand/right_joint_states.npy`: `(939, 6)` object-array joint states (available).
- `raw/timestamps/{timestamp,frame_id}.npy`: 553-frame camera timeline.
- `../assets/mesh/apple/apple.obj`: object mesh, loaded successfully.
- `../assets/robots/xarm_inspire_f1_right.urdf`: Inspire F1 robot geometry, loaded successfully.
- `C2R.npy` and camera intrinsics/extrinsics are present.

The HRDexDB geometry helper is `/home2/wyy/oyx_ws/HRDexDB/hrdexdb_contact_heatmaps/` at the same Git checkout. Its `generate_contact_heatmaps.py` and `hrdexdb_io.py` perform URDF FK, object-pose loading, mesh transforms, and nearest-vertex proximity. The generated heatmap semantics are unsigned nearest-vertex distance, clipped at 2 cm; they are a contact candidate/proximity signal, not force ground truth.

The larger DFTP episode tree has `raw/hand/force.npy`, tactile, and arm torque paths, but in the current checkout these files are Git-LFS pointer stubs (about 128–132 bytes). They cannot be used as measured wrench labels without fetching the gated large objects. The available `inspire_f1` subset likewise has tactile/torque paths as pointers while its geometry, poses, arm position, and joint states are readable.

### Existing Inspire simulator implementation

The checked-in `third_party/DExplore` source is Git `00d5c6c7a7eadf4237e0f808aad40943950c8165`. `dexplore/env/tasks/base_dexplore_task.py` acquires Isaac Gym net-contact-force and force-sensor tensors (`acquire_net_contact_force_tensor`, `acquire_force_sensor_tensor`, `refresh_net_contact_force_tensor`, and `refresh_force_sensor_tensor`). `dexplore/env/tasks/dexplore_inspire.py` defines the 18-DOF Inspire hand and its finger-coupling rules.

This is enough to construct a native smoke once Isaac Gym is available:

1. read per-body contact force vectors from `_contact_forces`;
2. read contact-body/world positions from the existing rigid-body state tensor;
3. transform contact points and force directions into each object COM frame;
4. use the CHORD friction-cone/support implementation to obtain a `(num_envs, num_bodies, num_basis)` support tensor;
5. compare CWS-on versus CWS-off in one short rollout.

The current DExplore contact tensor is a body-level net force. It does not by itself expose the exact manifold of per-contact forces that CHORD uses; this first smoke should therefore use each contact body’s net force direction as one representative normal/ray and report that approximation explicitly.

### SPIDER

The Task preflight card `docs/DATASET_PREFLIGHT.md` records a prior four-source SPIDER Inspire audit: source revision `44717007de41cbef7565dff7ff9f4453557a2d3d`, acquisition commit `03e23cc`, audit commit `a6c6c96`, MuJoCo 3.7.0 isolated environment. Its contract was `qpos=25, qvel=24, ctrl=18`, 100 Hz saved states, and fixed-identity mesh flow. The card explicitly records `missing_force_contact_labels=True` and `isaac_control_replay='NOT_TESTED'`. Thus SPIDER provides portable geometry and saved trajectories, while simulator contact recomputation is a separate label-generation step.

## Static CHORD support smoke

Command (read-only input, output under the current worktree `tmp/`):

```bash
python3 /home2/wyy/oyx_ws/HRDexDB/hrdexdb_contact_heatmaps/generate_contact_heatmaps.py \
  --dataset-root /home2/wyy/oyx_ws/HRDexDB/v0 \
  --asset-root /home2/wyy/oyx_ws/HRDexDB/assets \
  --mesh /home2/wyy/oyx_ws/HRDexDB/v0/assets/mesh \
  --hand inspire_f1 --object apple --scene 0 --frame-id 326 \
  --output tmp/hrdexdb-chord-audit-20261011/apple0-frame326.npz \
  --summary-json tmp/hrdexdb-chord-audit-20261011/apple0-frame326.json \
  --robot-visual-scope hand
```

The helper loaded one frame with 1,342,830 robot visual vertices and 36,745 object vertices. There were 28,554 robot vertices and 370 object vertices within 2 mm. Selecting 80 representative candidate object contact points, estimating object-mesh normals, constructing 8 friction-cone rays plus the normal per contact, and using `w=[f; p×f]` produced:

- wrench matrix: `(6, 720)`;
- basis: 12 directions (`±` six coordinate axes);
- torque scale `rc`: 0.0401266 m;
- support range: `[0.339589, 0.999999]`;
- all support values finite;
- self support reward: `1.0`;
- a deliberate support permutation reward: `0.92448`.

This proves that the available mesh/pose/URDF contract can feed the CHORD support computation and that the tolerance reward is numerically finite. It does not establish that the nearest-vertex candidate is a physically correct contact, nor that the recorded trajectory contains force labels.

## Cheapest next experiment

The least expensive simulator experiment is a one-environment, one-frame/short-rollout Isaac Gym probe using the existing DExplore Inspire task and one already audited SPIDER or HRDexDB scene:

- keep the current policy/action and simulator state fixed;
- collect `_contact_forces`, rigid-body positions, object COM pose, and episode frame index;
- compute the CHORD support tensor with the public geometry/math contract;
- log finite-rate, active-hand/body counts, support magnitudes, and CWS reward;
- replay the exact same rollout with the CWS term disabled and measure only contact-support metrics and object-flow error.

Use the force-closure gate if reference contact geometry is noisy. Do not treat the static HRDexDB proximity heatmap or SPIDER saved-state flow as a reference wrench. A native CWS-on/off performance claim requires the simulator contact tensors and time alignment to be recorded in the manifest.
