# Independently supervised implementation review

User authorized one read-only code-review subagent on 3 October 2026. The
review pinned commit881fb0d, compared22e47f5..881fb0d and inspected the shared
native/training substrate. No additional agent or Broker was created. The
original project, assets and existing checkpoints remained read-only.

## Confirmed engineering defect

Main-session diagnosis found `_apply_collision_filter` indexing rigid-body
names by rigid-shape index. Actual native Inspire asset has25bodies but only
13collision shapes. Seven wrist links and five fingertips have no shapes.
The existing loop assigned incorrect filters to four non-thumb intermediate
finger shapes (2instead of3) and thumb proximal base (3instead of2).

The fix applies the unchanged name rule through the SDK's actual body-to-
shape ranges, including empty and multiple-shape ranges. Independent
supervisor review found no new defect in this correction. A regression
executes the real method extracted from the current source against a real
IsaacGym actor, then queries native properties; it does not duplicate the
implementation under test. Native metadata only, zero simulation/model
steps, CPU chosen because no model or simulation computation is involved.

| Check | Before | After |
| --- | ---: | ---: |
| Native shapes with wrong filter |5|0|
| Regression process exit |1|0|
| Body / shape count |25 /13|25 /13|

Evidence: `ENG-20261003-inspire-shape-filter-before/results.json` and
`ENG-20261003-inspire-shape-filter-after/results.json` in the contact-response
output directory. Same asset SHA256:
`7d0023168a22191de2126d71f8d2a810f82e8cba56b74c82be6db04812b80f58`.
Source before `9824b67e3fdbe2dcc81302311cbb2694f50c07d0b48a0283852402f6d5babcb7`;
after `a3678db03da9f61384b0a8217cee86f8881e24c385aab967a8fd66c468c06033`.

Evaluation sets `args.test=True` internally, so the actual table filter is1.
The hand's2/3difference therefore cannot be dismissed based on the absence
of a CLI `--test` flag. Previous mesh/PD/neural audits did not query shape
filters: this defect was outside their coverage. Historical success labels
remain observations of the legacy physics; they do not establish performance
under the corrected environment. The defect's performance contribution is
not established by metadata correctness alone.

## Checks without a confirmed defect

The independent reviewer reconstructed current PRE decision observations,
eight-tick successor observations, held requests, all105tick task labels and
native goals on recent panels and selected historical panels. No mismatch
was found. SciPy Rotation supplied an independent quaternion implementation;
SDK successor coordinates agreed after float32 conversion, and object mass,
body-relative positions and quaternion norms showed no unit/frame mixup.

With physical-loss weight zero, encoder/task gradients agreed exactly with
task-only loss in coldQ/budgetQ checks. Frozen critics still supplied finite,
nonzero actor gradients. Stored optimizer counters matched1500critic and
1000actor updates (500for an intermediate checkpoint). Code identities
agreed. This was not an independent replay of every optimizer update and is
not an exhaustive absence-of-bugs guarantee.

One design limitation remains: actor Tanh followed by executor Tanh limits
effective requested amplitude to tanh(1)=.761594times the nominal scale.
Training and deployment share this convention, so it is not a confirmed
train/deploy mismatch. Most Gaussian collector requests lie outside the
actor's raw box; some successful collectors also do. Raw-box coverage does
not prove those final clamped PD targets are unreachable. No action-range
change or parameter sweep follows from that observation alone.

## Next decision

Run the prospective [one-panel physics-sensitivity Probe](../experiments/probes/P-20261003-inspire-filter-impact.md)
with retained655actors and all normalization/checkpoints fixed, no training.
Record actual hand/table filters in every native environment. A material
signal motivates corrected-environment substrate review, not a Cm benefit
claim. A small signal does not prove that corrected-physics training would
be unaffected. Preserve both physics histories without rewriting old data.
