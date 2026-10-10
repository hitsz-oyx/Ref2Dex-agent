# Task state: point dynamics pretraining

Updated2026-10-10. Branch `cm-pointflow-effect-pretrain`; execution remains on RLG.
Root Mission/Campaign apply. Video learning is an intermediate representation
route toward action-conditioned Cm, not a replacement for self-trained robotic
policy validation.

- Main sources: OakInk2/GRAB/ARCTIC,4,378,478 overlapping TRAIN anchors, retained
  sharedstats50k checkpoint. Weak video/tactile are outside that manifest.
- EPIC native hand-conditioned access: zero locally qualified H4+K24 hand windows
  after quality/split/clock fixes; hand/object labels cannot be silently promoted.
- [Video Probe](experiments/probes/P-20261010-video-point-dynamics.md): history-only
  PTv3 per-point head and weak-track loader are runnable. h24 initial500-step run
  fails the benefit gate; all dev endpoint object labels came from one clip.
  Window births improve train support but h24 still has only one supported dev
  clip. h8 recovery gives78/71 train/dev windows,4/2 supported clips. Fixed500-step
  h8 run at `6d7fdb2` gives model/static object clip-macro24.808/9.464mm. Both
  qualified dev clips fail; independent review reproduces metrics, input isolation
  and gradient masking. Recipe-level UNPROMISING; route-level UNCLEAR.
- [Tactile readiness](experiments/probes/P-20261010-egotouch-label-schema.md): two
  verified original TRAIN bundles and chest RGB,43/47 frames, relative clock
  agreement<1us/0.333ms.217/441 cells per hand are finite. Bend/tactile role and
  per-record normalization still need qualification.40-task contact flag search
  found allfalse; annotation semantics are UNKNOWN, not physical no-contact.
- Windows reverse SSH connection works; NAS has ample storage. Remote GPU driver
  mismatch remains outside this local campaign. [Connection guide](../../../../docs/user/连接远程服务器.md)
  records established access and limitations.

Current artifact group:
`outputs/cm-pointflow-effect-pretrain/video-point-dynamics-20261010-r1/`;
`train-r1/`, `train-h8-r2/` retain final checkpoints, progress and fit audits.
Video artifacts1.9GiB, within the5GiB Probe cap; models/fit used one free GPU1
and<4min total GPU wall time. No active training remains; no new large downloads.

Next decisions: identify whether weak3D pseudo tracks and noisy history velocities
are the limiting signal before adding updates/data; qualify pressure-only masks
and original normalization before a tactile auxiliary adapter. No scale-up or
native encoder-transfer run follows from the current negative probes. Additional
seeds and long runs are deferred evidence until a minimal positive signal exists.
No change to the final Mission claim, no policy training, no remote system work.

Additional tactile gate: official217-cell maps alias138/147 unique raw indices;
NPZ global max keys differ from the release converter's per-hand keys. Whole-
record normalization may expose future maxima in historical inputs. Use original
raw sensor values with fixed scaling/TRAIN-only statistics in any future adapter;
retain sparse validity, original normalization metadata and uncertain roles.


The observed-history OLS audit also loses to static in both qualifying dev clips
(15.014/6.763 and20.523/12.165mm), though it reduces last-two-CV errors. Next
video work should inspect pseudo-motion/observation quality rather than add steps.
The [channel-contract note](research/2026-10-10-egotouch-tactile-channel-contract.md)
now reproduces all finite/NaN values of both tactile samples within3e-8, confirmed
by root. Candidate groups/28 right-hand imputed cells are processing evidence;
hardware tactile/bend roles remain UNKNOWN. A raw/255 sensor representation can
be explored without making physical contact claims or future-max leakage.


2026-10-10 further progress: [raw sensor/hand conditioning](experiments/probes/P-20261010-rawsensor-hand-conditioning.md)
acquired10 tasks across5 settings,14.39MB/40 verified labels, and qualified330/535
complete windows from4 fitting/5 held tasks. Many original empty hand rows are
retained as NaN/invalid, not silently imputed. Fixed raw/255 matched500-update
three-arm MLP screen at `cc16f9e` fails: groupA h24 held-task macro actual future
hand4.607 counts vs history4.238/shuffle4.131/persistence3.435. Group hardware
roles remain UNKNOWN. Fit gains are real but no held-task teacher benefit. No
native auxiliary integration or expansion follows. Total local additions<120MB.

The rigidity audit is mixed (.393/.763 residual-to-motion ratios for qualifying
dev clips), so no uniform depth-noise attribution. Decision checkpoint after
three neural negative gates: prioritize current observation/physical interaction
context and weak-label quality before more updates; keep core Mission and mature
native main3 baseline. Next cheap action is paired RGB/context qualification for
new diverse raw-sensor tasks. No active training remains.


Paired-current RGB/context follow-up: [visual context screen](experiments/probes/P-20261010-egotouch-visual-context.md)
verified ten chest videos38.86MB, all original frame counts/IDs agree, max relative
clock deviation0.667ms. Root fixed30-frame inspection confirms visible hand/object
context; nominal Home/Office/Outdoor categories share the same physical table, so
no environment-independent claim. Frozen local ImageNet ResNet18 extracted3655
frame-local features,8.46s/3.24MB/389MiB CUDA. Learner uses only current start+3.
Six matched500-step MLP arms at347a184 finish10.72s/667MiB CUDA: groupA endpoint
macro history_rgb4.173/future_hand_rgb4.415 vs persistence3.435 counts. Both preset
context and conditional-motion gates fail; no sensor/native auxiliary scale-up.

Same-track h8 image-coordinate audit atdbcd471 retains exact3D denominators.
Dev last-two CV2.488/37.554px improves over image-static5.204/49.492px in both
clips; H4 OLS2.516/45.405px fails preset10%-in-both gate. This is a useful contrast
with3D history extrapolation, not proof of depth noise because camera motion and
world estimation differ. Next minimal weak-label action is a common endpoint-
camera reprojection diagnostic before choosing a new weak-video loss/label route.
All training/extraction ended;9 focused temporal/geometry/interface tests pass.
New artifacts remain well within campaign bounds; no new remote/policy operations.
