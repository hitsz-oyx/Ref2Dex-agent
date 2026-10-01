# P-20261002-executable-contact-opportunity

HF13 Decision Probe1/1, stage ENGINEERING, scientificslot0/1. Branch
agent/cm-executable-options. Frozen design:
[D-20261002](../../decisions/D-20261002-executable-contact-options.md).

Hypothesis: feedback expert or anchored pose-hold plans controlling the full
prediction window offer retention opportunities absent from sparse2+8control.
Decision: positive held-out local opportunity permits fitting a new plan-
conditioned Cm; insufficient/negative opportunity returns to grasp geometry/
candidate mechanics, no old Cm re-labeling or PPO/success-rate sweep.
Cheapest test is known-propensity real execution, no neural fitting yet.

Seven options: six frozen experts, recomputed from new native observations on
each of10ticks, plus hold the native-compatible independent joint targets
captured at decision time. Hold wrist commands invert relative native PD;
fingers invert native affine map and project couplings/limits. No extra closing
force or arbitrary ± perturbations. Six-tick cooldown remains. Old2+8Cm unused.
All10actual raw commands/native PD targets and object/contact/clearance labels
saved. The fixed anchor remains unchanged across the plan. Future MPC two-step
replanning must be tested separately from these actual full10labels.

Geometry: all25002vertices of declared airplane collision source mesh, native
scale, transformed against upper plane of actual thin-table mesh (localY,
world-up sign from table quaternion). Record signed min clearance; no256point
approximation. This is source-mesh/plane geometry, not exact PhysX VHACD or
identified pairwise contacts. Native hand/object net-force proxy retained and
named; no formal true-grasp claim from10ticks.

Engineering seed420/private8420, quotas2not-yet-clear +2already-clear windows
each env. Actual GPU0/1 (fresh admission;2/3 became occupied),96env,650ticks/240sec native budget. Tiny synthetic
inverse/rotation/full-feedback tests CPU (startup exceeds this tiny fixture).
Require hold PD/online expert execution, full nonterminal10labels, native
geometry orientation/finite values, frozen checkpoints. No utility inference.

Science fixed421–432/private8421–8432,96env/seed,650ticks,per-episode quotas
8not-yet-clear +8already-clear. Already-clear prestate means center>=3cm above
fixed rest AND full-mesh plane clearance>=2mm. All triggers require3consecutive
force-proxy ticks and a full nonterminal H10. No deletion of partial labels:
any partial/terminal window invalidates collection. Two preassigned sampling cohorts: env0..63 follow base until an already-clear
state before any intervention; env64..95 can contribute both general and
already-clear states. Both retain per-stratum8window quotas. Allocation after
observing state: general8slots uniform1/8; already-clear slots0/1/2/3/5 each.04,
base slots4/7 each.20 andhold6.40. Actual base propensity.25general/.40clear,
hold.125general/.40clear, other five.125general/.04clear. Record the entire
allocation vector and cohort; merge only the identical base4/7rules.
Do not merge distinct feedback programs merely because first commands match.
Randomized baseA/baseB null estimates population variability, not an individual
solver-repeat oracle. Each actual record belongs to observed pre-action state.

Frozen split across alltwelve panels: initial (motion,start) group SHA256 of
`9851/{motion}/{start}`, first8hex modulo100: <50fit,50..69cal,>=70held.
No initial group/episode cross split. Fit alone chooses bestfixed option by
IPW mean clearance-supported signed retained-height change on already-clear
states, requiring fit>=24matches/eight episodes/four initial groups peroption;
base always comparator. Independent held evaluation: primary hold vsbase,
secondary frozen bestfixed vsbase (both reported, no individual oracle/regret).
Primary score is minimum last3 center height above rest, floored0, times joint
last3 force and all last3 mesh-clearance>=2mm, minus positive starting height.
Report original supported-height/drop labels, contact fraction, all-window
retention and already-clear lost-clearance (<2mm on anytick) separately.

Screen: held each compared side>=48matches/12episodes/8initial groups,
clear-supported score uplift>=max(2mm,2*abs(baseA-baseB null point difference)),
frame-group AND episode-cluster90% lower bound>0, lost-clearance risk excess
<=2pp and last3 joint-force retention deficit<=5pp. Confidence resampling uses
1000draws/fixed9852. This is an exploratory screen, risk point bound not proof
of safety. Either predeclared primary or frozen-secondary passing makes local
opportunity PROMISING; both sufficiently supported failing UNPROMISING; needed
support missing UNCLEAR. Missing/null one-slot support remains explicit.
No post-outcome threshold/split/seed/checkpoint selection. No full success metric.

Budget engineering+science<=60min/8GiB,oneidleGPU preferred; collect twelve
phases serially. No unknown process kills/external writes/overwrite; input,
source,asset,expert and motion hashes pinned. Failed engineering charged.
No scientific run starts until source-matched engineering passes. Actual
elapsed/storage/physical support and all failed gates archived. C3OPEN.

Engineering bootstrap: first launcher failed Python parsing before creating
any output/native/GPU process (missing quote, code41dcf96). Corrected before
smoke data; actual graspenv interpreter compiles allthree runtime files.
Recorded.0276s, conservatively charge.03s in engineering cumulative budget;
no scientificslot consumed or data overwritten.

Engineering r1 terminalFAILED before native launch: GPUs2/3 became occupied
by external jobs. No worker/data created. Preserve manifest, charge elapsed;
r2 uses newly confirmed idleGPU0/1, no process interference.

Before scientific data, engineering r2 provided217complete windows/94episodes,
35already-clear cases with quotas2, alltarget/feedback/geometry contracts pass.
Even optimistic quota scaling makes six uniform panels fragile for48held
matches perprimary side. Amend sampling before science: twelve freshpanels,
64clear-first envs plus32general envs; common-prestate targetedhold/base.4/.4
andotherfive.04each. Keep original support/gain/risk/split gates unchanged.
R3 must reverify the new allocation/cohort contract. R2 stays COMPLETED with
its original codeidentity and allcost charged; no scientific outcome inspected,
scientificslot0/1. Expected engineering+collection20–30min, capstill60min.
