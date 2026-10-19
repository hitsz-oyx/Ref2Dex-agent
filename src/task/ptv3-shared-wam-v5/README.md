# PTv3 shared WAM v5

v5 predicts24 frames at30Hz with **21 semantic landmarks per hand**,32 original
object-flow queries, and2048 object-surface contact queries. It encodes2048
object points and up to778 surface points per hand with serialized PTv3 at5mm.
The shared Transformer receives256 scene +42 hand +32 goal tokens, not778
trajectory tokens. There is no KNN, MANO reconstruction, FK or contact-label
generation in the model or training reader.

Current preparation/check results and live status paths are in [docs/STATE.md](docs/STATE.md).

The independent package is `src/ptv3_shared_wam_v5`. It reuses existing PTv3
primitives through `paths.py`; on the remote server set
`REF2DEX_V5_BASE_CODE_ROOT` to the frozen v4 training-code root. v4 models,
checkpoints, source data and running training are not edited.

## Representation

Joint order is wrist, then thumb/index/middle/ring/little, each with three
articulated origins and a tip. Human thumb roles are CMC/MCP/IP/tip; other
fingers are MCP/PIP/DIP/tip. Robot landmarks are geometric semantic proxies.
The recorded Shadow asset uses palm as root, thbase/thhub/thdistal/tip for the
thumb, and knuckle/middle/distal/tip for other fingers. The old Shadow11
scaffold placed its thumb-base slot at thhub (MCP), whereas the human first
thumb point is CMC; v5 fixes that role instead of duplicating the coincident
thhub/thmiddle origins. The old landmark audit therefore compares that legacy
point to v5 thumb slot2, not slot1.

MANO sources retain their original fingertip vertex convention and coordinate
transforms. Robot surfaces are deterministic samples of the recorded collision
primitives, not an invented MANO mesh. Surface samples carry side and16-part
semantics, without requiring robot and human vertex IDs to correspond.

`C` has17 channels: one contact confidence and16 part probabilities (palm plus
three segments per finger). Labels are **offline nearest-surface-vertex
proximity**, with confidence1 inside3mm, exponential10mm falloff, and zero
beyond33mm. They are not measured force/contact, exact triangle distances, or
CEDex's normal-oriented capsule formula. Part labels use dominant MANO skinning
joint or actual robot collision-link semantics. Contact supervision is masked
if an expected hand lacks reconstructed geometry, and on synthetic HOI4D
midpoints. Missing geometry never creates negative labels.

Following [the user proposal](docs/user/ref1.md), the next decision is restricted
to data and C. Before changing the teacher or BCE, run the two independent audits
in [the contact-definition probe](docs/experiments/probes/P-20261019-v5-contact-definition-audit.md):

- `tools/audit/audit_contact_prevalence.py` streams completed caches and reports
  per-source P(c>0), P(c>0.25), P(c>0.5), soft mass, valid denominators and16-part
  positive counts. It also counts TRAIN/DEV K24 label appearances under uniform
  windows within source, using the reader's repaired H4 admission and reporting
  rejected windows (hands are inferred from a bound contact contract). This
  latter view does not apply the motion-stratum or
  source sampler; it must not be presented as the exact optimizer distribution.
  `--watch` waits for newly completed caches within a declared deadline and
  publishes explicit partial coverage until every planned sequence is audited.
- `tools/audit/audit_contact_normals.py` compares a bounded, TRAIN-only panel
  with half uniform and half contact-enriched valid frames. Its candidate adds
  a backside penalty using existing object normals; it is explicitly not a
  CEDex reproduction. Baseline regeneration must exactly match cached labels.
  Reports include invalid normals, changed queries and a nearby opposing-face
  proxy; selected3D comparisons are saved as PNG/NPZ for visual review.

Object normals are not yet certified outward on every source/CAD. Discrete
MANO vertices and robot primitive samples also retain different surface-density
errors. These are teacher limitations to inspect, not evidence that a new
formula is correct. A low positive rate (e.g. below2%) is a trigger to examine
balanced BCE against the actual sampler, not an automatic loss switch. Merely
increasing uniform surface sample count does not itself reduce positive fraction.
The audits keep the17-channel C contract, joint21 horizon normalization, model,
existing labels, and PPO unchanged. No CEDex grasp data is admitted.

Tau and C remain parallel predictions of the observed state and goal. There is
no trajectory/contact consistency loss or embodiment-specific retargeting in
this implementation. BC/RL executors have not been ported to joint21 here.

## Data preparation

The2026-10-19 remote workspace is:

```text
/mnt/ugreen_nas/storage/oyx_storage/ptv3-shared-wam-v5-20261019-r1/
  data/plan/            # frozen admitted v4 TRAIN/DEV window identities
  data/packets/         # compact original annotation packets and readiness records
  data/geometry/        # deterministic2048-point CAD surfaces
  data/manifest.json   # complete2747-object surface registry
  data/hands/           # audited per-sequence joint21 and surface caches
  data/contacts/        # per-frame contact confidence/part/validity caches
```

Tools in `tools/run/` implement these stages:

1. `export_preparation_plan.py` freezes admitted windows; no held-out TEST.
2. `export_hand_packets.py` extracts original local annotations. DexYCB uses
   its calibrated45-component MANO PCA basis; H2O uses native camera transforms.
3. `sync_preparation_inputs.py` transfers packets before their readiness records.
4. `prepare_hand_overlay.py` reconstructs surfaces/21 points and validates the
   original overlapping scaffold to50µm. Surface offsets relative to the wrist
   use float16, checked for at most0.1mm quantization error. Joint positions
   remain float32. Data is stored once per frame, not per overlapping window.
5. `export_object_surfaces.py` samples original CAD triangles, streaming the
   UniDex archive once and retaining all original32 flow identities.
6. `prepare_surface_contact.py` computes labels offline and prioritizes an
   initial cross-source sample. Its KD tree is absent from all training paths.
7. `freeze_dataset.py` publishes a separate manifest/index and recomputes
   TRAIN-only joint21/flow moments. `--require-complete` rejects pending
   sequences; a bounded `--max-sequences-per-source` snapshot is explicitly
   marked as an engineering subset. Dataset admission and subset scope are
   recorded independently.

The final recipe preserves v4's source probabilities and main3 motion-stratum
weights0.6/0.2/0.2 (renormalized over existing strata); native windows remain
uniform within source. Categories are recovered by exact original window
identity, not recomputed from new labels. The initial nine-sequence engineering
GPU snapshot predates this sampler binding and used uniform windows within
source; its timings are feasibility checks, not a matched training comparison.

Writers publish atomic files/directories and explicitly set readable task-cache
permissions to handle the NAS's observed mode000 files. Run manifests preserve
the Python/PyTorch environment, code revision, paths and bounded runtime.
Preparation stops below20GiB free space or at its declared deadline. A `STOP`
file in a stage output requests a reversible stop; completed data is retained.

## Model checks and training

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python -m pytest -q \
  src/task/ptv3-shared-wam-v5/tests
```

The current768-width/12-head/12-layer config is
`configs/pretrain_joint21_surface2048.json`; the earlier1024-width config is
retained at `configs/pretrain_joint21_surface2048_scale1024.json`. The contact head uses
checkpointed256-query chunks to bound activations; equivalence and gradients
are tested. Losses use separate global DDP denominators for joint motion,
motion-weighted object flow, contact BCE and positive-contact part CE.

The contact head now also reads the PTv3 feature of the **same object point ID**
for each of its2048 queries. This retains the256-token shared trunk and adds no
spatial search. The encoding identity is `object-centric-surface-contact16-local-id.v5`;
old global-only checkpoints cannot silently resume into this architecture.
DEV reports separate root/root-relative, old11/new10, fingertips, contact-phase,
per-horizon hand/flow errors, contact MAE in broad/core proximity bands, and a
16-part confusion matrix. Contact phase is a whole-object horizon diagnostic,
not a per-hand attribution; confidence1 approximates the3mm band after float16
quantization. These diagnostics do not add new losses.

`tools/run/train_pretrain.py --smoke --steps 8 --batch-size 64` runs bounded
real-data AdamW updates and writes timing/results only, with no model checkpoint.
It requires explicit `--config`, `--manifest`, `--stats`, `--output` and
`--git-commit` paths. Full training additionally requires an admitted manifest
and an allocated `--seed`. DDP, source-macro DEV evaluation, per-rank RNG restoration
and recent-checkpoint rotation are implemented. Four-rank CUDA startup passes;
CPU stop/resume matches uninterrupted optimization exactly, while bitwise CUDA
resume remains unverified. v4 checkpoints/statistics cannot silently resume as v5.

`tools/audit/run_gpu_smoke.py` supports the user's authorized temporary pause of
the identified WAM process group. It checks process/GPU ownership, installs an
independent resume watchdog, and restores training in `finally`.

Each training/smoke run uses a fresh output directory. `--resume` reads the
previous checkpoint into a new directory so checkpoint rotation only touches
files created by the current run. First-step and configured milestone
checkpoints are protected from recent-checkpoint rotation. `hand_in_ptv3=false`
now excludes both joint and surface points from geometry aggregation; the
explicit hand branch remains separately controlled.

## Verified data correction from ref1_1

The [second user audit](docs/user/ref1_1.md) was checked against the actual remote
environment. Trimesh4.12.2 capsules are already centered: the previous extra
half-length translation did shift Shadow surfaces. The corrected constructor
centers using measured local bounds and validates MJCF radius/length. FK joints
alone could not detect this surface error. Old Shadow surfaces/contacts remain
diagnostic artifacts and are excluded from the repaired recipe.

Repair writes `data/hands-r2/` and `data/contacts-r2/`; unchanged human sequences
and completed human contacts are explicitly inherited by symlink. Source4 is
always rebuilt. The actual geometry registry is **`data/manifest.json`**, with
NPZ assets under `data/geometry/`; tools take `--geometry .../data`.
H4 admission now uses a shared predicate requiring a persistent semantic joint
across all four history frames, identical to the reader. Final indices and
TRAIN moments are recomputed after repair.

Cache runs carry reconstruction/dependency/upstream contracts and refuse
incompatible resume. New arrays receive a publication-time payload identity;
contact also binds hand metadata and geometry payload identities. Frozen
recipes and readers pin upstream contracts and per-sequence metadata. Explicit
legacy inheritance is limited to human caches from the controlled old run.
No per-batch array hashing is added: direct manual mutation of immutable array
files without updating metadata is outside this lightweight guarantee. Full
payload rescanning is not claimed. Partial geometry registries are rejected.
Continuous triangle/primitive teacher distances, per-hand C attribution,
joint/surface voxel separation and v4 model-only initialization remain separate
follow-up decisions; this change does not assume their benefit.

`tools/run/finalize_preparation.py` waits for hand/contact stages and freezes the
final manifest and TRAIN statistics automatically. Incomplete inputs remain
explicitly marked as a partial subset; it never launches training.

## Four-GPU sustained training

The user-requested mixed run is recorded in the
[training probe](docs/experiments/probes/P-20261019-wam-v5-mixed-pretrain.md).
`complete_contacts_parallel.py` takes over a confirmed stopped serial stage
with the identical teacher and input binding. `finalize_preparation.py
--training-admitted` only admits complete hand/contact coverage. Sustained
training rejects engineering/partial recipes and requires all nine sources
through `tools/run/launch_pretrain.py`.

The launcher owns four GPUs and an isolated torchrun group, records GPU state
and progress, and applies runtime/free-space boundaries. `STOP` saves the last
completed update before exit; resume uses a fresh directory and the exact same
data/config/world size. Checkpoints carry optimizer/per-rank RNG/elapsed time
and DEV selection. Recent rotation protects the first, milestones and best.
Loss logs are reduced over all ranks; the first DEV panel runs at20 and then
every1000, using a fixed seed171 panel. The four-rank engineering startup now
passes; the CPU recovery contract test does not claim bitwise CUDA/DDP resume.
