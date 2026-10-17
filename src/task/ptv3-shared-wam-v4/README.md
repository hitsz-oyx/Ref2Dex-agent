# PTv3-Shared-WAM v4

v4 keeps the audited v3 input, geometry path, 24-step hand/scene forecasts and
`G_task` conditioning, and adds an explicit interaction head. The intended
planning chain is:

```text
G_task -> local 24-step goal/scheduler -> WAM -> (tau, C) -> executor
```

The new `C` output is object-centric and embodiment-light, with shape `B,O,24,8`:

1. soft contact likelihood;
2. hand/object proximity in metres;
3. object-centre minus hand-centre approach vector (metres, 3);
4. relative velocity (metres/second, 3).

`C` is derived from the existing hand and object future labels by
`interaction_target`; no new corpus is silently introduced. The prediction is
returned in physical units (`interaction`) and fixed-scale units
(`interaction_normalized`). Invalid object slots are masked to zero. A future
Ref adapter can later produce `C_ref` and fuse it before the robot executor; Ref
is intentionally not in this WAM core.

This is an architecture/probe implementation. It does not claim robot benefit
until a matched `tau-only` versus `tau+C` executor experiment is run. It reuses
the v3 nine-source reader, while using a new schema
`ref2dex.ptv3-shared-wam.v4` so v3 checkpoints cannot be loaded accidentally.
