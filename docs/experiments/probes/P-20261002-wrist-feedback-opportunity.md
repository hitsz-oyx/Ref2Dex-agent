# P-20261002-wrist-feedback-opportunity

HF14 Decision Probe1/1, engineering COMPLETED; frozen science starts. Owner current session, branch
agent/cm-executable-options. [Frozen design](../../decisions/D-20261002-wrist-anchored-feedback.md).

Test whether continuing expert finger feedback with an anchored wrist offers
local retention opportunity beyond full base feedback. Candidate6 uses base
fingers;7 cup fingers;0..5 existing experts. All execute ten new-observation
feedback ticks, then reobserve with6tick cooldown. This fixes the pure-pose
freeze mechanism, without arbitrary action perturbations or extra gripping
force. Old Cm/V/PPO/actors remain unchanged.

Known support5760/5760 normalized-positive, airborne0/480 positive; audited
force units before collection. Each new window saves actual raw link/object
forces, mass/gravity/substeps/mode, normalized proxy plus legacy.1N proxy,
whole source-mesh/plane clearance, first eight candidate commands, all actual
commands/native PD, and all pre-step native expert observations. Native GPU
replays frozen experts on saved observations and checks selected finger action;
terminal audit independently checks force labels/relative-wrist PD/geometry/
propensities. This is net-force presence proxy, not identified hand-object pairs.

Engineering seed440/private9440,96env650ticks/quota2perstratum. Native240sec,
parent290sec. Science seeds441–452/private9441–9452, same environment and
quota8perstratum, already-clear cohort64/general32. All complete nonterminal
H10; no trimming. General nine-slot uniform; clear probability
[.04,.04,.04,.04,.20,.04,.20,.20,.20], duplicate base4/8 merged to.4; each
anchored.2; other experts.04. Private allocation after observing prestate.

Frozen initial-group split SHA9851 fit<50/cal<70/held>=70. Fit chooses one
anchored variant (>=24matches,8episodes/4groups); held comparison vsbase needs
>=48matches,12episodes/8groups per side. Primary signed clearance-supported
last3normalized-contact retention gain>=max(2mm,2*abs duplicate-base null
point); initial-group and episode90% cluster-bootstrap lower bounds>0;
lost-clearance risk point<=+2pp, last3joint-proxy point>=−5pp. Null and both
variant estimates reported, no held winner selection. Qualified fit-best-fixed
among all8 reported separately; family classification follows primary chosen
anchored comparison. Missing support UNCLEAR. No individual oracle/regret claim.

<=60min/8GiB cumulative engineering+science, own one freshly admitted GPU0/1,
immutable inputs/expert/asset/motion checks. Stop on input drift, invalid
execution/partial labels, occupied devices or scope budget. Positive permits
new plan-conditioned physical Cm; negative closes family without threshold/
seed tuning. No full-task success or RL-learning conclusion from this Probe.

Engineering105windows/53episodes,46alreadyclear,25anchored windows. All25
anchored finger programs vary;80expert windows vary. Frozen-expert replay,
raw-force ratio/label replay, relative wrist PD and whole-mesh geometry all0
error. Complete nonterminal labels and merged propensities verified; both own
PIDs exited.112.350sec/10.533MB. [Audit](P-20261002-wrist-feedback-engineering-record-audit.json).
