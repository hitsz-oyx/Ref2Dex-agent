# Rigid transport candidates have useful oracle capacity

P-20261003-rigid-transport-capacity-r1 COMPLETED/PROMISING at69cc802.
All three prospective gates and independent audit pass. This is a capacity
result with outcome-fitted scalar/endpoint selection, not a learned or
deployable Cm result, contact identification or policy utility.

## Evidence that changes the next decision

Reuse384held whole corrected655episodes,6144fixedwindows, same seed and four
fixed own-policy arms previously examined. No new collection, network fitting
or policy optimization. Command/current-state actuator trained only on the
disjoint384train episodes remains unchanged. Same64canonical airplane points.

Anchor is previous-point-motion persistence. State-only oracle chooses an
outcome-fitted scalar in[0,1] toward either zero motion or exact rigid inertia.
Full family adds13visual-link endpoints, each transporting the current
object by Hnext*inverse(Hcurrent). Causal Hnext comes from predictednextq;
measured-hand Hnext is an explicitly unavailable diagnostic. Both full
families include exactly the SAMEtwo state-only segments. All families use
each window's real nextobject motion to select endpoint ANDscalar deliberately.

| Oracle family | Episode-mean64point EPE/mm |
| --- | ---: |
|State-only, outcome-fitted|0.368403|
|Causal hand transport +state, outcome-fitted|**0.294084**|
|Measured-nextq hand transport +state, outcome-fitted|0.294313|
|Raw persistence, no outcome fitting|0.629368|
|Raw rigid inertia, no outcome fitting|0.626504|
|Zero motion|0.660944|

Causal versus equally outcome-fitted state-only gain20.173368%; near-hand
gain20.506746%. Causal/measured-hand ratio0.999219906 satisfies execution-gap
gate. All3 gates true, prospective PROMISING. Measured and predicted hand
families do not contain each other: their slight error ordering does NOT mean
the actuator is more accurate than measured motion. It follows from these
different candidate fields with oracle scalar/endpoint selection.

Predeclared diagnostics, all equal-weight available episodes:

| Group | Windows/episodes | State oracle/mm | Causal oracle/mm | Measured-hand oracle/mm | Raw persistence/mm |
| --- | ---: | ---: | ---: | ---: | ---: |
|Motion0|2048/128|0.014566|0.014356|0.014192|0.116740|
|Motion1|2048/128|0.526253|0.469905|0.466551|0.982374|
|Motion2|2048/128|0.564391|0.397990|0.402198|0.788989|
|Near hand|2124/256|2.363974|1.879200|1.905687|2.823930|
|Far from hand|4020/384|0.064233|0.063419|0.063340|0.519530|

Near is inherited current-query unsigned distance<2cm, not contact. Subsets
are not window-weighted decompositions of primary. All actor-arm reports
retained. Larger hand-capacity signal is in near states andmotion2; no
subgroup changes the fixed primary/gate classification. Motion0 remains
unsolved in actual policy tasks despite small oracle prediction error.

Causal oracle chooses zero endpoint4222windows, rigid inertia161, one of
13hand links1761 (28.662%). Selected coefficient quartiles0/.038381/.646537/
1/1. These labels are NOTobserved contact states: many endpoint choices can
explain similar flow, zero-flow and inertia attenuation account for much
overall predictability, and oracle choice is label-fitted. A learned model
must face these degeneracies without access to the outcome.

## Representation contract

This is a union of15one-scalar segments with the SAMEpersistence anchor,
not the full multi-contact convex hull. Each hand endpoint is a rigid
transport; a mixture MEAN point flow need not produce a realizable rigid
pose. Choice among allvisual links is permissive and not contact-feasibility
validated. Therefore useful capacity does not establish an executable
physical contact law, force closure, impulse model or identified intervention.

Each segment minimizes mean Euclidean point error with1e-9m norm smoothing,
52float64derivative bisections and boundary signs. Segment winner uses true
EPE. Oracle coefficients, selection and all predictions are retained. The
state-only comparator has the same outcome access and scalar fit; the positive
gap is not simply relative to an uncalibrated inertia baseline. Adding
candidates increases oracle flexibility; whether deployable input identifies
the useful choice remains the next uncertainty.

## Independent evidence and cost

All6144rawheld rows/split/timestamps and predictedjoint states independently
rebuild exactly from corrected source and inherited actuator. All6144state-
only point/pose fields rebuild with SciPy, max4.440892e-16m. Firstfixedwindow
per384held episode receives independent13link geometry reconstruction, max
9.084761e-9m. Other5760hand geometry rows are NOTindependently reconstructed.
FiveSDKbody origins max5.919861e-5m (0.059mm); source/full native audits
inherited by hash. Independent endpoint rotation orthogonality max2.442491e-15.

ALL196608segment constraints/predictions/errors receive independent NumPy
convex supporting-line gap certificates; max2.399880e-9m including smoothing
bound, below1e-7m. Independent SciPy bounded optimization re-solves12288
segments in384selected windows, maxobjective gap3.180814e-15m. Storedsegment
error max3.469447e-18m; all primary/per-parent/subgroup/baseline/gates recompute
exactly (metric max0). Every full-family window error<=state-only error;
both full families' two state endpoint banks are exactly the state-only bank.
Read-only supervisor independently reaggregates and rebuilds all saved
segment predictions/error/winners and superset checks; no blocker.

Wall56.656120s,353030828bytes, within300s/512MiB. CPU engineering3.215s,
freshly idleGPU6capacity9.614s, CPU independent audit38.218s. No new native
ticks or neuraloptimizerupdates.196608scalar solves are deliberate offline
label fits, not neuraltraining and not a learned policy. Protected inputs
unchanged; parent609124 and children609183/609625/610081 verified absent.
Float64world baseline differs slightly from inheritedfloat32normalized
persistence0.629373mm; discrepancy5.25e-6mm is within fixed1e-3mm audit bound,
not a normalization or metric change selected after fitting.

## Action and remaining requirements

Keep the rigid-transport family for ONEbounded causal-learnability Probe.
Use the retained wholetrain episodes to learn coefficients/endpoint scores
from current geometry/state and command-derived transports; evaluate the
same family on held episodes WITHOUToutcome-fitted coefficients or futureq.
The state-only learned model must receive the same training/held split,
calibration budget and outcome availability. Fix controls, architecture,
normalization, update count and gates prospectively before any fit.

Do not connect this label-fitted oracle to an actor or claim knowledge-policy
utility. Only a learned-information gate permits fresh randomized corrected
native qualification before matched policy-training utility. Same-seed
viewedholdout, object/tasks/morphology generalization, distinctiveness and
formal multi-seed policy Validation remain unresolved. Mission unchanged,
goal ACTIVE, journal NOTREADY.
[Decision](../decisions/D-20261003-after-rigid-transport-capacity.md).

Evidence: contact_response/output/P-20261003-rigid-transport-capacity-r1,
capacity/{fields.npz,*_oracle.npz,results.json},audit.json,logs,runmanifest.
[Prospective card](../experiments/probes/P-20261003-rigid-transport-capacity.md).
