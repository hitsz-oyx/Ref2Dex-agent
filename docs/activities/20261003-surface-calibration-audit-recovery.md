# Preserve fits, recover the independent audit

Run r1 at d47e1fe reached four completed1200-step fits, then failed in the
independent audit BEFORE any classification was admitted as valid evidence.
NumPy combined advanced indexing `measured[:, body_ids, :3, 3]` reordered
the result to5x16x3; SDK labels are16x5x3. This was an audit implementation
error, not a training/input tensor error. Original traceback and FAILED
manifest remain untouched; no neural checkpoint or data is overwritten.

Correction selects bodies first, then translation, and explicitly checks
shape. It changes exactly this audit statement and no tolerance, feature,
target, source split, head/encoder, optimizer, checkpoint or prospective gate.
Run r2 inherits all r1data/fits by fixed SHA and symlink, only repeats the
previously failed independent CPU audit. Zero new model/optimizer/physics
work;4800actual updates remain attributable to r1, not doubled. r2 guards
all retained source inputs/artifacts/code, verifies ancestor PIDs absent and
allows exactly the declared audit-only edit compared with committed r1code.
Budget<=900s/1GiB (inherited aliases counted), within current campaign.
Results remain provisional until all original independent audit checks pass.
