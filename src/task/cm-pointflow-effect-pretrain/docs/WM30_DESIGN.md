# OakInk2 WM30 / K24 implementation contract

User architecture and confirmed choices govern this run: program anchor plus
0.5m current-center local objects; three independent matched arms H, H+A,
H+shuffle(A), plus evaluation shuffle of H+A; all627 annotation sequences,
maximum4GPUs and24hours wall time for the training group. No Policy/Evaluator/Y.

30Hz is fixed raw frame-ID phase0 modulo4 (120Hz source), no interpolation.
History is [t-3,t-2,t-1,t] at30Hz and future is [t+1,...,t+24]. Reject gaps,
invalid poses and steps crossing raw adjacent tracking jumps. Single hands
remain masked in right/left11-point slots. Keep observed distant hands.

Anchor candidates are objects mentioned in program intervals anywhere in that
sequence. Current program activity informs the interaction stratum, not scene
features or object inclusion through future labels. Local object selection uses
current centers only, distance<=0.5m, including anchor. Current normal/velocity,
three past displacements, target flag, historical-static flag, point/hand/keypoint
and local object instance IDs enter H. No future-derived flag enters H.

Use one fixed C=inverse(T_anchor,t) for ALL history/current/future geometry.
For object m define current local pose S0=C*T_m,t, future Sτ=C*T_m,t+τ.
Relative effect Fτ=Sτ*inverse(S0) acts on current local points: Pτ=Fτ*P0.
Translation supervision is Fτ.translation; evaluation additionally reports
object-center displacement error to distinguish rotation-around-origin effects.
Rotation uses6D Gram-Schmidt. RMS radius weights geodesic loss. All losses
are means over valid objects/times/points, lambda_R=lambda_P=1. No dense residual.

512 deterministic canonical mesh surface samples plus normals/object; no RGB.
Sparse stem:1cm voxels, subvoxel residual plus continuous motion features,
SubMConv3D64/128/256,5x5x5 voxel patch mean pooling,384-dimensional spatial
patch tokens,8 Transformer layers/6heads/FFN1536. No KNN or spatial hard gate.
Action MLP+time/hand/keypoint embeddings,4Transformer layers at384; dynamics
6TransformerDecoder layers (query self-attention and scene/action cross-attention),
24 queries per local object. Whole action chunk is visible without causal mask.
Object query features use current center/orientation/radius/local ID and target
flag. Every local object is supervised; historical static features are not GT future.

Sampling60/20/20: anchor has motion over0.8s (>2mm OR>.02rad at any future
step); small-effect anchor with current program activity OR current sampled
surface proximity<5cm; small-effect anchor without those interaction cues.
Categories are sampling/metrics metadata only, never model inputs. Report
natural and balanced held-out metrics; include static baseline. No2cm gate.

Split at sequence level using deterministic hash/seed210:80%train,10%val,
10%test. Save subject/task/object overlap audits; do not claim unseen-subject
or unseen-object generalization. Validation fixed panels with no random-frame
mixing; final test remains separate from model selection. Training seed211;
shared initialization, sample schedule, optimizer and updates. Shuffle uses
whole24-step chunks and masks from donors in the same hand-presence group,
with no fixed points; donor current-relative and step deltas come with donor.
Report that this is observational future-hand supervision, not causal action.

AdamW3e-4, weight_decay.01, clip_grad1, bf16 where backend permits, warmup
then cosine decay. Select batch/update count from measured full-architecture
smoke throughput, preserving all architecture widths/layers. Three independent
single-GPU arms use GPUs0/1/2, no more than4GPUs during preparation. Group
24h deadline, equal fixed updates derived conservatively from slowest throughput;
checkpoint resume includes optimizer, RNG, step and input/config identity.
Bounded artifacts<=70GiB additional,>=10GiB free reserve, latest/best/final
only with atomic replacement of this run's own latest; never overwrite external
checkpoints. Stop on nonfinite loss/gradient, unsafe data or group deadline.

Decision experiment: does actual A improve held-out moving point EPE vs H and
shuffled A? Cheapest useful answer is this user-authorized single-seed matched
full-corpus run. Probe labels only; positive signal motivates later formal
Validation/robot adaptation, weak signal requires implementation/data review.
