# OakInk2 100-sequence data readiness result

Engineering audit completed; annotation-only OakInk2 supports the requested
bimanual semantic-hand/SE(3)-effect data extraction. Predictor learnability,
causal action utility and policy transfer remain untested. No pretraining/PPO
has started. [Frozen protocol](OAKINK2_DATA_PREFLIGHT.md), user [ref2](ref/ref2.md).

## Identity and resources

- task: cm-pointflow-effect-pretrain; branch: cm-pointflow-effect-pretrain.
- run_id: oakink2-data-preflight-20261006.
- GPU audit commit:4bf2f7f6d26ecf015486d16c93e670dbe08094ca; CPU cache-summary
  commit:57b4262. input_manifest.json records frozen source/protocol/MANO hashes;
  postprocessing confirmed every hash unchanged. Frozen protocol stays intact.
- Dataset revision:21705616140d726607027e70d58b7837f442ffd8.100 fixed evenly
  spaced sequences from627, plus five asset archives;105 files checked against
  published SHA256 and byte sizes, acquisition6289339102bytes/429.32seconds.
- One GPU (CUDA_VISIBLE_DEVICES=0), audit877.62seconds within1200second cap.
  Raw/extracted/cached run size7224314936bytes within12GiB. No RGB acquisition.
- Artifacts: `../../../../artifacts/cm-pointflow-effect-pretrain/oakink2-data-preflight-20261006/`
  relative to this file's directory. Contains download_manifest.json,
  input_manifest.json, audit/result.json, audit/continuity.json and NPZ caches.
  Artifacts are ignored by Git; machine-local outputs remain available here.

## Data checks and distributions

646565 hand frames, all with both hands observed in this selected subset;
477 sequence-object pairs,4965774 object frames. Zero frame-ID gaps, invalid
rigid poses or missing meshes. Missing-hand support is implemented but this
subset does not empirically test missing-hand examples. Wrist reconstruction
agrees with raw translation, and analytical object point/effect correspondence
maximum error is4.44e-16m. These checks prove extraction consistency, not an
independent physical ground-truth alignment validation.

World-space geometry follows official MANO/transform conventions in meters;
object axis extents range5.86mm–487.56mm.97.03% of program windows also meet
5cm sampled-surface proximity, supporting practical hand/object alignment.
Semantic-point proximity is a coarse diagnostic, not a contact label.

At120Hz, h8 means66.67ms. Hand-point h8 displacement p50/p90/p99 is
2.29/18.95/51.51mm. Raw object translation p50/p90/p99 is
0.066/0.893/15.97mm; rotation is0.00101/0.01404/0.09148rad.
Adjacent hand-point displacement p99 is6.49mm, object translation p99 is
2.17mm, object rotation p99 is0.02384rad.

Diagnostic motion = h8 translation>2mm OR rotation>0.02rad:

| Selection | Valid overlapping windows | Moving | Static fraction |
| --- | ---: | ---: | ---: |
| All valid | 4961958 | 476229 | 90.40% |
| Program | 731579 | 384679 | 47.42% |
| Near hand | 1077578 | 422808 | 60.76% |
| Program AND near | 709842 | 382481 | 46.12% |
| Program OR near | 1099315 | 425006 | 61.34% |

98/100 sequences have at least one moving interaction window. Counts are
sequence-object windows, strongly overlapping; they are not independent
samples or effective training-set size estimates.

## Continuity exceptions and next decision

Object transforms are rigid but not universally continuous.11 adjacent
translations exceed5cm and353 adjacent rotations exceed0.2rad, spread across
62 sequence-object records. Their union affects1622 valid h8 windows,
including1344 program windows (0.184% of program windows). There are also2
hand-point steps above5cm. These are diagnostic flags; large rotations can
include tracking/pose ambiguity or real motion and need inspection, not an
unsupported claim of annotation corruption. Frame pairs are saved under
object_jump_locations in audit/continuity.json. Current counts above include
these flagged windows; no silent cleaning was applied.

Decision: retain OakInk2 as the sole first pretraining source. Use program
manipulation windows first, keeping both hands and masks; use proximity as a
quality check. Review/mask windows crossing anomalous steps and balance
moving/static windows before a bounded predictor Probe. Split by sequence
(and inspect subject/task/object overlap), never random overlapping frames.
Preserve120Hz h8 explicitly; any horizon change is a new experiment choice.
The next useful step is a small held-out predictor comparison with a static
SE(3) baseline, before committing to large training. Full627 annotations are
36.65GB; expansion is reasonable after this readiness gate, but has not been
started. No extra datasets are needed to resolve this first question.

## Verification and recovery limitations

New OakInk2 source/document checks pass;100 real sequences and cached summary
finish successfully with frozen hash checks. Git metadata was recovered as an
independent local repository from remote history; current working files were
preserved, branch unchanged. Recovery snapshot60824b8 retains unpublished
Task file contents, but not their lost original commit identities.

The broad recovered-history verification reports102 passed/6 failed. One old
current-policy-value test requires deleted motion outputs; the other five
reference a URDF below the deleted Ref2Dex-agent-baseline directory. Targeted
reproduction confirms missing-path failures. This audit does not use those
inputs. Deleted historical checkpoints/rollouts/SPIDER runtime files remain
unrecovered; retained documentation cannot substitute for replayable artifacts.
No remote push was performed.
