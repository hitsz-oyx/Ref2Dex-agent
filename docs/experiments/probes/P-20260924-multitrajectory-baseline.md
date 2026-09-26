# P-20260924-multitrajectory-baseline

- Classification: Decision Probe.
- Branch: `agent/grab-multitrajectory-baseline`.
- Work version: `multitrajectory-baseline-probe`.
- Cm: off.

## Question and decision

Can one self-trained actor continued from the s3 airplane e260 checkpoint
make useful held-lift progress on a fixed 12-motion, ten-object corrected
mixture while retaining airplane performance? If several object identities
show held-lift and airplane does not collapse, expand the converted GRAB pool
and design a matched Cm-on/off intervention against this baseline. If the actor
collapses or most objects remain at zero, first inspect motion sampling,
reference contact semantics and curriculum rather than attributing failure to
Cm. The cheapest informative test is a single-seed e260→e300 continuation,
followed by per-motion first-episode assessment.

This is a local converted subset, not the full GRAB dataset. The study does
not establish generalization, stable multi-seed performance or Cm utility.

## Fixed run contract

Use `src/task/CmResidual/configs/multitrajectory_12_motion_probe.json` and
`run_multitrajectory_baseline_probe.py`. Each listed motion appears once;
hard-object oversampling is off. Source checkpoint SHA256 is
`16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.
Train seed70, 64 environments, horizon32, endpoint e300, one idle GPU,
wall time <=60 minutes, output <=5 GB. Stop on source/input drift, occupied
GPU, nonfinite training or missing endpoint checkpoint. Run status and
scientific outcome will be recorded separately below.

## Results and next decision

Training `agent_multitrajectory12_s70_e300` completed on commit `748939e`
in 265 seconds; all 12 motions loaded. At e300, new evaluation seeds201/202
gave **16/128 held-lifts (12.5%)**: 6/30 airplane, 7/10 toothpaste,
2/12 alarmclock, 1/12 cup, and zero on apple, cubesmall, duck, mug,
phone and waterbottle. Mean contact fraction was 0.264. Full per-motion
report: `outputs/Dexplore/agent_multitrajectory12_s70_e300/analysis_e300_s201_202.json`.

Matched source e260 checkpoint evaluations on the same 12 motions and
seeds gave **13/128 held-lifts (10.2%)**: 10/30 airplane, 1/10 toothpaste,
1/12 cubesmall and 1/12 cup, zero on the other six objects. Thus the
40-epoch continuation shifted success from airplane toward toothpaste
and alarmclock; the pooled +3/128 is too small to treat as an efficacy
finding. This single training seed, two evaluation seeds Probe is
`UNPROMISING` as a broad multi-trajectory baseline. It does not prove
that more data, longer training or Cm cannot help.

Next cheapest decision test: evaluate existing self-trained train5 e320
and balanced e360 checkpoints on the same motion mixture. If their
per-object strengths complement e260/e300 enough to cover at least four
identities with positive held-lift and improve the object-oracle upper
bound substantially, test a policy-option Cm that predicts future
contact-supported lift. Compare its routing against object-only and
shuffled-Cm routers with the exact same expert portfolio. If portfolio
coverage remains narrow, focus on baseline training/curriculum before a
new online Cm policy attachment. These choices are autonomous under the
user's 2026-09-24 Probe authorization.

### Portfolio check

The fixed train5 e320/e360 checkpoints were evaluated on the same 12 motions
and seeds201/202, with no new training. Totals were e320 **18/128** and e360
**21/128**. Across four self-trained actors (e260, mixed12 e300, train5 e320,
balanced e360), the best per-object count chosen *after seeing both evaluation
seeds* is 32/128, an optimistic oracle upper bound. At least 3 successes on
an identity occurred only for airplane, mug and toothpaste; apple, phone and
waterbottle stayed 0/10 under every actor. A simple object router selected on
seed201 and tested on seed202 scored 12/64; reversing the seeds also scored
12/64. This is only 24/128 versus the best single actor's 21/128 and does
not warrant a new Cm router over these four weak, largely object-specialized
policies. The original "positive on four identities" criterion was too weak
because it counted isolated 1/10 events; the actual per-object record shows
the limit. This route is `UNPROMISING` for broad multi-object grasp or a
high-value Cm candidate portfolio. Preserve the portfolio results as a
diagnostic, and use the already successful single-object expert portfolio
as the next Cm decision substrate.
