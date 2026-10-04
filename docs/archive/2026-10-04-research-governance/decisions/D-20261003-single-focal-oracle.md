# Single controlled episode oracle, fixed background

GPU mixed replay fails and persistent PD storage does not change it. CPU
physics likewise fails mixed isolation. Do not attribute failure exclusively
to GPU or continue backend/tolerance scans. r1's valid TRAIN trajectories and
four6000-update Q fits remain reusable; its mixed-deployment oracle interpretation
is invalid. No confirmed source-code bug behind this semantic mismatch.

Choice: test one controlled episode at a time in a full scene with11P0
background instances. All alternatives affect ONLY that controlled hand;
other hands use P0 feedback. Selected alternative is re-executed from frame0
with the IDENTICAL whole-scene action protocol, not merged with independent
choices for neighbors. Same-world baseline replay already proves that this
can reproduce native histories, but engineering and all selected futures
must pass before any oracle utility claim.

Fresh seed763,12envs (4/reference), all12designated subjects. Original eight
805options, decision36/H32, nominal corrected GPU PhysX, own frozen P0,
persistent PD tensor. Short queries end at68 and have NO terminal success
label. Reuse frozen r1Q models/TRAIN-only normalizers by SHA; no refit, new
seed/model/horizon/option tuning, or r1EVAL-label selection. Their training
was under global-option96-env scenes; new fixed-background12-env test has
a documented scene/behavior shift. Same shift applies to all four arms.

84nonzero short queries +baseline; actual202tick deployment for each unique
selected subject/option, reused across arms only when exact commands coincide.
All12subjects/4arms/full105mesh criterion; original5pp joint-vs-state/effect
and>=P0 gates, strict query/actual tolerances unchanged. Each subject has its
own full-scene rollout; simulation cost includes all background instances.

Engineering <=300s/512MiB; then ordinary Probe <=3600s/6GiB, one idle GPU for
physics/inference, zero NEW optimization steps (6000 inherited separately).
Stop on same-world future mismatch rather than relaxing oracle criteria.
Positive is bounded headroom only, negative is frozen selector/option/protocol
only. No universal Cm refutation, mathematical upper bound or journal claim.
