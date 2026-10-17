# WAM-v4 architecture contract

v4 keeps the audited v3 goal-conditioned trunk and its 24-step hand/scene
forecasts. It changes the interaction output from a three-anchor nearest-point
summary into a dynamic object-centric graph:

```text
G_task + current 3D history
          |
          v
   shared PTv3/WAM trunk  -----> tau_hand[24], tau_scene[24]
          |
          +---- hand tokens x object query tokens x time
                                  |
                                  v
             C_HO[T, hand-point, object-point, channels]
```

The graph has shape `B,O,T,2,11,P,9`, where `T=24`, `P=32` in the current
data contract, and the nine channels are contact confidence, object-relative
surface position (3), radial surface normal (3), normal approach velocity and
tangential slip speed. Every edge is masked by the corresponding object point
and hand trajectory validity. This preserves the relation between a specific
hand point and a specific object patch; it is deliberately not mean-pooled into
a single WAM feature.

The labels are generated from the existing 32 object query points. Contact is
a Gaussian of hand/object distance and normals are radial because dense surface
normals are not present in the v3 nine-source contract. A dense surface view can
replace the target builder without changing the graph channels or executor
interface.

The planned reference path is separate from v4 pretraining input: a demo
encoder will later produce structured phase/contact/trajectory tokens, and WAM
will use those tokens for local continuation. The current implementation does
not concatenate a raw reference or claim robot-control benefit. The first
decision probe remains a matched `state + tau` versus `state + tau + C_HO`
executor test after the graph head passes its data and warm-start audits.
