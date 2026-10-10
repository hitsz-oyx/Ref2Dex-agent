---
schema: ref2dex.probe.v2
probe_id: P-20261010-decoder-prefix-fitting
experiment_id: P-20261010-decoder-prefix-fitting
date: 2026-10-10
task: trajectory-policy
branch: main
git_commit: pending
claim_id: C3
hypothesis_family: HF-trajectory-policy-action-space
probe_index_in_family: 2
seed_pool: probe
seeds: [297]
decision_changed_if_positive: test optimized c reconstruction before enlarging decoder
decision_changed_if_negative: replace straight temporal segments before actor initialization
status: UNCLEAR
run_id: decoder-prefix-fitting-20261010-r1
---

# Can fitting c preserve the immediate wrist trajectory with fixed D?

## Motivation / Decision Note

Decision for independent H->c trajectory policy initialization, which serves
self-trained manipulation and later matched Cm policy utility. Last goal turn
was progress: completed native coverage and excluded promoting the copied-knot
initializer/frozen-R combination (dense4/4 versus compressed0/8 terminal holds).
It did not establish an optimum over c or refute48D trajectory policies.

Choose the cheapest representation audit before more simulation or learning.
The existing D is linear in world XYZ knot positions, even though c parameterizes
them by tanh in the current object frame. Unconstrained least-squares knots are
a superset of bounded c positions; their residual gives a conservative lower
bound on reproducing the selected geometric wrist path. This is not a lower
bound on control success, learned-policy return, full-hand error or all motion.

If even optimal straight-segment XYZ cannot closely reconstruct the executed
prefix/nominal FF target, repair temporal representation first. Otherwise test
a fitted initializer in a separately declared native Probe. No next experiment
is authorized by passing geometry alone; no WM/actor training in this card.

## Protocol

Freeze same four knots[1,8,16,24],24future frames and8executed frames; all68
windows of verified hand-derived geometric q. No recorded future robot q/object,
commands or force inputs. Build B24x4 by exact piecewise-linear interpolation.
Build V from one-sided future endpoints and central interior differences at
dt1/30; exclude live q0. F=B+.1V is the actual wrist PD feedforward base map.

Compute separately optimal prefix position and optimal prefix FF residuals by
`numpy.linalg.lstsq`, with no bounds/regularization. Then one joint initializer:
equal prefix8 position and FF squared errors, plus tail16 position errors with
residual weight.25. No coefficient/node/dimension sweep or outcome selection.
Report prefix3D RMS and maximum, normal-equation residuals and per-window bounds.
The exact projection optimality applies only to the two separate objectives;
the joint initializer must not be called a simultaneous optimum of both minima.

Fixed engineering closeness screen: every window separate position bound<=5mm
AND separate FF bound<=10mm; if either exceeds its band, local UNPROMISING for
close reproduction within this D, not a physical failure theorem. If all bounds
and actual joint fit pass, local PROMISING for testing initializer; if separate
bounds pass but joint fit fails or bound/finite/provenance checks fail, UNCLEAR.
Bands diagnose trajectory fidelity; they were not validated as grasp thresholds.

Convert joint-fit XYZ knots to c, keep original rotation/finger oracle knots,
decode on one idle GPU2 for FK coverage. Audit encode/decode knots, velocity
operator, finite values, native limits/coupling and original input hashes. Do not
alter D or oldTask code, and do not use the original recorded q as a label.
Save c/fit/bounds/actual decoded tau and a first-window plot.

## Resources / stop

One CPU pure NumPy projection/array audit and one GPU2 FK batch; no neural
training/inference, simulation, new data, extra seed or checkpoints. <=120s
wall/2GPUmin/64MiB new artifacts. CPU least-squares is tiny4-column statistics;
GPU handles batched rigid FK per Campaign policy. Stop on source drift,
nonfinite values, geometry provenance mismatch or unsafe GPU occupancy. Stay
main; no new branch/push. Seeds[297] identify inherited Probe context; solver
is deterministic and consumes no RNG/validation seed.

## Results

Not run yet.

## Limitations / future evidence

Single hand-derived motion; matching its wrist path is sufficient for one
initialization goal but not necessary for successful alternative policies.
No c control optimization, finger/rotation optimization or separately intervened
FF dynamics. Positive requires full native behavior verification then actual
H->c training; even that is not matched Cm-on/off benefit.
