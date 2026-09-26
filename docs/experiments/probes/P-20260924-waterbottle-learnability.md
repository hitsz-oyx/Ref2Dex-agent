# P-20260924-waterbottle-learnability

Date: 2026-09-24. Branch: `agent/cm-cross-object`.
Classification: Decision Probe — shared-policy interference versus task learning.

## Question and fixed decision

Object-balanced train5 continuation improved toothpaste but left waterbottle
at 0/13 held-lifts and zero positive H10 support, while the read-only official
diagnostic actor reaches 64/64. Can the same self-trained e320 checkpoint and
unchanged reward learn waterbottle when multi-object gradient interference is
removed?

Materialize a one-sequence, hash-pinned waterbottle training root. Resume the
pre-balance train5 e320 checkpoint for 40 epochs to e360, seed179, Cm-off,
with the original reward/curriculum and no held-out data. Evaluate 64 first
episodes on seed178 and export transitions.

If held-lift reaches >=16/64 and the unchanged H10 support gate passes, mark
waterbottle individually learnable and treat shared multi-object policy
representation/interference as the next blocker. If either condition fails,
mark this short unchanged-reward route `UNPROMISING` and redesign curriculum
or initialization before constructing a larger Cm. Do not add epochs or tune
reward on the observed seed. This is not a generalization or Cm utility test.

One idle GPU, <=20 minutes training plus <=10 minutes evaluation, <5 GB.
Stop on source/split/checkpoint drift, GPU conflict, non-finite training or
missing output.

## Primary result

Status: `UNPROMISING` for the fixed 40-epoch single-object continuation.
Training and the 64-environment e360 evaluation completed, but held-lift was
0/64, mean hand/object contact fraction only 0.0169, mean maximum
contact-supported lift about 0.01 mm, and there were zero positive H10
load-bearing windows. The >=16/64 and support gates both failed. Do not add
epochs or tune the same reward on this evaluation.

This rejects the simple claim that removing multi-object gradients is enough.
It does not yet distinguish a poor e320 waterbottle initialization from
catastrophic degradation during the 40 continuation epochs. Before changing
curriculum, evaluate the already-fixed source e320 and saved intermediate
e340 checkpoints on the identical waterbottle-only seed178 protocol. If e320
has contact fraction >=0.10 or >=32 positive H10 windows and a later
checkpoint loses at least half that support, diagnose continuation collapse;
if e320 is already below both thresholds, diagnose state-initialization/task
mismatch. An e340-only improvement followed by e360 collapse would motivate
an early-stop/stability change. These checks use existing checkpoints and do
not reopen the failed e360 gate.

## Checkpoint diagnosis

The fixed e320/e340/e360 replay established continuation collapse. At e320,
waterbottle had contact fraction 0.131, 138 positive H10 windows spanning 14
environments, and passed the support gate despite 0/64 final held-lifts. At
e340, contact fell to 0.043 and positive windows to zero; at e360, contact
fell further to 0.017 with zero positive windows. Thus the source actor does
visit useful transient states, but unchanged PPO rapidly destroys them rather
than converting them into sustained grasp.

The next route should not add training epochs. For Cm representation data,
freeze e320 and collect equal per-object rollouts so sparse environment
allocation does not hide available states. For policy learning, any later
Cm-on/off Probe must test whether the representation prevents this collapse;
plain continuation is not an adequate control substrate by itself.
