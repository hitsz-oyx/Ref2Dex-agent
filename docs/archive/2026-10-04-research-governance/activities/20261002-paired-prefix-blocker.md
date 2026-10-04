# Paired common-prefix matching blocker

Paired option601 executes completely under code7ad72a8. Full native/P0/options/
PD/mesh audit passes; classification UNCLEAR because common-prefix matching
fails. Baseline label agreement186/192 (96.875%) passes. Initial robotq/dq/
object state is exact, but prefix object positions differ up to18.833mm,
robot XYZ4.404mm, joint angles.164rad. The hypothetical oracle88/192 versus
P063/192/duplicate67/192 cannot be interpreted as same-state action utility.
All owned phases terminal,156.338s/249505391bytes, inputs unchanged.

Fast deterministic saved-trace reproduction:

```bash
/home2/wyy/miniconda3/envs/graspenv/bin/python scripts/diagnose_paired_option_prefix.py \
  --source src/task/CmResidual/research/contact_response/output/P-20261002-paired-option-task-opportunity-r1 \
  --output src/task/CmResidual/research/contact_response/output/ENG-20261002-paired-option-prefix-diagnosis-r1 \
  --require-match
```

It ran2.419s and correctly fails `saved common-prefix matching gates FAIL`.
The initial table XY differs up to15.26µm. Native robot/object differences
already occur after FIRST physics tick, before intervention; first commands
are identical. Hand root is identical; actual body poses and DOFs diverge.
The first diagnostic included decision-tick commands, which are interventions;
the separately corrected decoder excludes those commands while retaining the
decision PREstate. This changes no primary gate, data or physical execution.

Ranked falsifiable explanations: per-actor SDK gains/mass/material differences
would appear in an all-actor property snapshot; environment-origin numerical
effects predict equal properties but spatially dependent arithmetic; hidden
initial solver/rigid-body cache state predicts initial tensor equality without
first-step state equality. No unique cause established by reading source.

Next cheap engineering check reads all768 actual SDK actor properties and
origins AFTER reset, with ZERO physics steps/model forwards. Unique
`ENG-20261002-paired-property-snapshot-r1`,605metadata seed, <=180s/128MiB,
same owned-child/GPU/protected-input guards. No scientific endpoint or matching
limit changed. Original601data and diagnosisr1 remain immutable.
