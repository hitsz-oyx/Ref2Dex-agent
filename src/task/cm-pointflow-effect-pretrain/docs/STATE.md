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
