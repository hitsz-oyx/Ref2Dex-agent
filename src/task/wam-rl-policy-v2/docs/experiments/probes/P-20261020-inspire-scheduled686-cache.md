# P-20261020-inspire-scheduled686-cache

This probe regenerates WAM conditions for the audited measured teacher corpus.
The task path and scheduler state come from each sequence's same-execution
teacher object rollout and `TaskSpec`; the old cache is read-only geometry
binding only.  The v4 checkpoint is frozen and has no trained CONTACT/RELEASE
event input, so this run is the v5-compatible spatial-goal baseline.

The output is flat, row-addressed `tau` (264 values) and validity arrays. Two
workers use disjoint sequence indices (`index % 2`) on GPU6 and GPU7. Invalid
workspace rows remain zero with validity false and are excluded by the later
scheduled BC admission. This is a cache generation probe, not a control or
grasp-success result.
