# Causal rigid coupling: capacity does not become matched learned advantage

P-20261003-rigid-coupling-learnability-r1 completed at code348302a,
UNPROMISING under all four prospectively fixed gates. The previous positive
oracle capacity screen is retained; this result tests actual learned
coefficients and endpoint scores without future-outcome input.

| Predictor | Equal-episode held EPE, mm | Near EPE, mm | Far EPE, mm |
| --- | ---: | ---: | ---: |
| Full15endpoint learned Cm |0.558129272|2.596552938|0.509577725|
| Matched learned2state-endpoint control |0.448095692|2.722162720|0.093955756|
| TRAIN-outcome-shuffled full |0.731395116|3.544529887|0.446884857|
| Previous-point-motion persistence |0.629367707|2.823930168|0.519530074|

Full improves persistence11.319% and shuffle23.690%, passing these gates,
but is24.556% worse than learnedstate-only, failing required10%gain. Near
gain4.614% also fails required5%; no rounding or subgroup rescue. Near uses
unsigned current-query distance<2cm, not a measured physical contact label.
Whole6144windows/384episodes have16windows each; near2124windows/256episodes
and far4020/384 weight available episodes equally, so their displayed EPEs
must not be window-weighted to reconstruct the primary number.

All three models have23874trainable parameters, commoninit4201 and shared
4202schedule,1500AdamW updates each (4500actualtotal). State-only has the
SAME99current state/geometry/command/aggregate-hand-motion features; it is
not action-blind. Its2endpoint tokens repeat15calls with balanced loss.
All120feature normalization statistics use TRAINonly. Shuffled controls
re-solve coefficients against permuted TRAINworldflow, not another row's
coefficient labels. Train384/held384whole episodes are disjoint,6144rows each;
this previously viewed same-seed corrected655panel remains diagnostic.
No new physics, actor learning, checkpoint selection or held coefficient fit.

Independent audit: all12288raw/split/state rows;768fixed feature/FK13transport
checks and5SDKbody checks (other11520handgeometry rows not independently
rebuilt); all184320TRAINown/shuffled scalar convex certificates; ALLheld
NumPy neural outputs, winners, flows, metrics and gates. Maximum rawfeature
delta1.90723e-6, transport8.48834e-9m, SDKorigin5.91986e-5m,
certificate2.21062e-9m, neural9.53674e-7, metrics0mm. First3optimizer updates
perarm replayed independently in NumPy,9total; maxparameter5.38166e-7,
loss8.88171e-8. Remaining4491updates not independently replayed. Read-only
review independently reaggregates all saved parent/subgroup metrics, verifies
commoninit/schedule and causal feature boundaries; no concrete blocker found.

Main244.564910s/286423899bytes, smoke4.710330s, GPU6prepare93.152102s,
GPU6fit25.664346s, CPUaudit109.590975s. Protected1455input hashes unchanged;
parent629727/children629802/630034/632163/632837 absent after completion.
All fits, optimizer states, schedules, early states, raw fields, labels,
features, predictions and audits retained.

## Separate fixed-weight decision diagnosis

After the negative gate, prospective saved-array decomposition card was
committed atc84f7bd before execution. P-20261003-rigid-coupling-error-
decomposition-r1 completes in3.441455s,0optimizer/native updates. It uses
future outcomes explicitly to locate error, never as deployable evidence.

| Substitution on the fixed full model | Overall mm | Near mm | Far mm |
| --- | ---: | ---: | ---: |
| Actual learnedcoef/learnedchoice |0.558129272|2.596552938|0.509577725|
| Learnedcoef, true-error-minimizing choice |0.486082987|2.123665858|0.485368010|
| Outcome-fitted coef, actual learnedchoice |0.359357410|2.146914278|0.113487344|
| Outcome-fitted coef and choice |0.294083832|1.879200099|0.063418581|

Best selection with the SAMElearned coefficients still loses to matched
learnedstate-only0.448096, failing the fixed10%whole-panel diagnosis gate.
Near diagnostic gate passes but cannot rescue it. Overall selection regret
0.072046286mm versus coefficient gap to fulloracle0.191999155mm; these are
saved-field decompositions with path dependence and interactions, not
independent causal mechanisms or contribution percentages. Independent
read-only reaggregation of every diagnostic value and all11source hashes
also passes with zero discrepancy. State-only
candidates from fullmodel also yield0.605847mm (best-choice0.548969), worse
than the separately trainedstate-only model. A score-only repair therefore
has insufficient capacity with these saved coefficients on this panel.

Close this bounded independent coefficient/own-error-confidence recipe.
Preserve the positive rigid-family oracle capacity and learnedstate control;
a separate state-anchored/direct-flow-loss design is the next decision input,
not an approved model or positive scientific claim. No density, steps, seeds,
width or subgroup rescans. Current scalar blended flows need not define a
single rigid pose, and endpoint winners do not establish contact legality.
No native/actor investment or formal journal conclusions follow this failed
learned gate. Goal ACTIVE, matched Cm policy-training utility NOTDEMONSTRATED.
