# OakInk2 annotation-only readiness audit

User ref2 switches current route to OakInk2 only: observed human hand keypoint
future trajectory conditions object effect. No SPIDER merging or predictor/PPO
training during this phase. Keep both right/left hands, fixed semantic IDs and
frame masks; absent hands are not interpolated or inferred from the other side.

Decision question: can annotation-only data provide aligned, nontrivial,
interaction-rich h8 supervised windows before large pretraining? Engineering
Blocker audit, not a causal intervention or policy-utility claim. First verify
100 fixed sequences, then decide corpus scale/filtering.

## Frozen inputs, protocol and boundaries

- Branch: cm-pointflow-effect-pretrain; prior checkpoint60824b8; audit script
  hash/run Git identity recorded before execution.
- Dataset: kelvin34501/OakInk-v2, revision21705616140d726607027e70d58b7837f442ffd8.
  Official annotation inventory627 sequences/36650885771bytes.
- Select100 evenly spaced indices from the lexicographically sorted pinned627
  paths, including endpoints. Not random-frame sampling. Total5941117662bytes.
- object_raw/repair/affordance + program/extension tarballs, no RGB/data/*.tar.
  Total acquisition6289339102bytes. Every file must match published LFS SHA256
  and size. Download deadline1800s, selected-byte cap12GiB and10GiB free reserve.
- Artifact root explicitly declared:
  /home2/wyy/oyx_ws/ai_ws/Ref2Dex-agent/artifacts/cm-pointflow-effect-pretrain/oakink2-data-preflight-20261006.
  Original outputs common-directory link became unavailable during this phase;
  recovery does not move or duplicate the new data.
- Licensed existing MANO assets at
  /home2/wyy/oyx_ws/Ref2Dex/dataset/arctic/data/body_models/mano, READ ONLY.
  Legacy chumpy/NumPy aliases set only in audit process, no environment mutation.
- GPU0 for batched MANO fixed-geometry forward/distance reduction, one worker;
  audit<=1200s/GPU-seconds, raw+processed total<=12GiB. Stop on nonfinite MANO,
  analytical correspondence failure, unsafe/mismatched data or deadline.
- Hand order right then left. MANO21 ->11 fixed IDs
  [0,1,5,9,13,17,4,8,12,16,20]: wrist+five MCP+five fingertips.
  Same center_idx0/flat_hand_mean/quaternion mode as official visualization;
  world joints = centered MANO joints + per-frame tsl, no camera conversion.
  Verify reconstructed wrist equals raw translation. Distant observed hands
  remain valid: program activity is not hand observation validity.
- Missing/nonfinite/zero-quaternion hand frames masked. Window requires8
  consecutive frame increments, at least one observed hand and unchanged
  per-hand presence over all9 frames; no frame-gap stitching or interpolation.
- Official mocap120Hz, therefore h8=.066667s. This is not previous30Hz h8.
- Target object mesh: repaired align_ds if available, otherwise raw align_ds.
  Fixed512 area-weighted canonical surface samples/object, deterministic seed
  derived from object ID. Object points = T_world_from_object * canonical.
- Future effect E_k=T_(t+k)*inverse(T_t), k1..8. Verify sampled h8 points from
  direct future transform equal E8 acting on current points (<1e-8m).
  Object rigid validity: orthogonality/determinant errors<1e-3, homogeneous
  bottom-row error<1e-6. Missing pose frames excluded from entire9frame window.
- Describe h8 object center displacement/relative rotation and hand displacement
  distributions. Diagnostic moving means translation>2mm OR rotation>.02rad.
  Program window requires all9frames within target object's annotated
  manipulation intervals (left/right union). Near window means at least one
  observed semantic hand point within5cm of sampled object surface on any
  window frame. Dense distance reduction is an audit, not a KNN network.
- Report raw/filtered static fractions, missing geometry and validity/continuity;
  high static fraction triggers interaction filtering before pretraining,
  rather than mixing datasets or immediately extending model training.

Runtime artifacts: download_manifest.json, dataset_info/anno_tree/root_tree.json,
input_manifest.json, dataset/{object_*,program*}, audit/result.json and cached
hand trajectories/object transforms/canonical points. CPU pickle/mesh processing,
GPU geometry; no learned predictor fitting or RGB requirement.
