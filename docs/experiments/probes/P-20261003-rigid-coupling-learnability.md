# Learn causal coefficients inside the unchanged transport family

Decision Probe. Previous goal turn is progress: rigid transport capacity
passes all3gates/audits and is archived in supplement-r16. Capacity coefficients
use outcomes, so they cannot be deployed or support policy utility. Mission
unchanged. This screen distinguishes useful causal coefficient/score inference
from an oracle-only span. Success permits fresh randomized corrected-native
qualification before actor training; failure closes this bounded inference
recipe without new budget/seed/architecture scans to rescue it.

## Fixed data and architecture

Inherited corrected655 wholeepisode split:384train/384held,16fixedticks each,
6144windows each, previously examined same-seed heldpanel. Same64canonical
airplane points,13visual endpoints+2state endpoints and persistence anchor.
Retained disjoint-train command/q/dq actuator. Build only missing TRAINcausal
fields and oracle coefficients (including a shuffled-outcome control), reusing
existing HELDfields exactly. No new simulation, task/success labels or actor.

One shared per-endpoint MLP120->128ReLU->64ReLU->2 (23874parameters).
The first output sigmoid gives w in[0,1]; secondsoftplus estimates log1p of
the point EPE/mm produced by its OWNcurrent predicted coefficient. Winner
minimizes predictedscore; decode a SINGLEendpoint A+w*(E-A), unchanged union
of15segments. No soft endpoint blending, family expansion or outcome fit at
evaluation. Blendedpoint flow remains a mixturemean, not necessarily a rigid
pose. Candidate selection remains permissive, not contact-feasibility proof.

Each causal feature token has99common+21endpoint components:

- q,dq/30,senttarget-minus-q,18each,54total, fixed joint scales translation.1m,
  wristrotationpi and measuredfinger upper-minus-lower.
- Currentobject translation/.1 plus firsttwo rotationcolumns,9total.
- Previousobject translationincrement/.01 plus world rotationincrement-I
  flattened/.05,12total.
- Previouspoint-flow fitted rigid twist,6total.
- Mean13currenthand-link poses in currentobject frame (xyz/.05+rotation9),12.
- Mean13causal hand-endpoint fitted twist,6total.
- Perendpoint currenthandrelativepose12 (zeros for state endpoints), its
  fittedtwist6 and kindonehot3 (zero/rigidinertia/hand).

Twist is meanobject-localpoint flow/.01 and least-squares angularfield/.05
using the fixed64point centeredlever Gram+1e-12I. Features use only CURRENT/
PREVIOUSobject, currenthand/currentcommand and causalpredictedhand transports.
Futureobject and measurednextq are never feature/selector inputs. Train-only
mean/std over6144x15rawtokens, stdfloor.01, common across ALLarms. No labels
or held statistics in normalization. Source/frame contracts inherited bySHA.

## Three matched arms and training

1. Full: all15endpoints, trainownoptimal coefficients.
2. State-only: only2inertial endpoints, replicated as [0,1]*7+[0] to15token
   calls. Common99features include exactly the SAMEjoint, geometry, command
   and aggregatepredictedhand information. No action-blind input handicap.
   Relative/currentmotion endpoint features remain their actual state values.
   Componentweights15/(2*replica_count) give equal state-endpoint lossweight.
3. Shuffled: full15endpoints, fixed permutation4203 ofTRAINworldnext-point
   flow targets. Oraclecoefficients are solved consistently against those
   shuffledtargets using the CURRENTrow's samefields/anchor, not copied from
   another row. Confidence targets use the same shuffledflow. No held shuffle.

All commoninitseed4201; outputweightszero, biases[-2.1972245773,.5413248546]
(initialw=.1,score1). Shared1500x32window schedule seed4202 with replacement,
6144rowbank, AdamWlr3e-4, weightdecay1e-4, gradclip10.4500actualupdates,
48000windowdraws/720000componentforwardrows perarm; the latter are correlated
derived labels, NOT720000independent transitions. Same normalization,
parametercount, actual update/window/component budgets and inputs.

Trainloss: component-weighted mean of
(w-w_oracle)^2 * min(meanpointnorm(E-A)/.01,1)
plus (softplus(score_raw)-detach(log1p(actual_predicted_EPE_mm)))^2.
Coefficient supervision is train-outcome-fit only. Directionweight suppresses
degenerate/nearconstant segments without arbitrary winner classification.
Confidence label follows CURRENTpredictedw; detach prevents score from
changing its own label through w. Evalselector never consults real EPE.
No fine-tuning/sourceprior, extra loss scan, epoch sweep, early stopping or
held-checkpoint selection. Save model/optimizer/commoninit/schedule/alllosses,
first3parameter states, all heldcoefficients/scores/winners/decodedflows.

## Fixed primary/gates/diagnostics

Primary equal-weight episode-mean64point EPE/mm, all6144heldwindows.
Full<=.9rawpersistence; Full<=.9learnedstate-only;
Full<=.95shuffled; nearFull<=.95nearlearnedstate-only.
PROMISING iff allfour; UNCLEAR iff persistence+state-only bothpass but notall;
otherwise UNPROMISING. No subgroup override. Near is inherited unsigned
current-query distance<2cm, not actualcontact. Report all motion/actor/near/far
and zero baseline, weighting available episodes equally in eachsubset.
No representation gain substitutes for matched policy-training utility.

## Audit and resource scope

CPU4row/3step smoke: independently reconstruct causal features; perturb
futurelabels/nextq and verify input/deploy invariance; independentNumPy
gradients/AdamW reconstruct tiny updates including detached confidence target.
CPUappropriate for tinyengineering, main data/NNfit/inference on onefreshly
idleGPU(prefer6). CPU independent file/geometric/statistical/NumPy audit.

All12288rawrows/split/timestamps; allstate-onlyfields/objecttargets and shared
train-only normalizer independently checked. FIRSTfixedwindow per384TRAIN
and384HELDepisode:768independent SciPyfeature/FK13transport/5SDKbody origin
checks, other11520handgeometry rows not independently reconstructed.
Features atol2e-5/rtol2e-6raw; transport2e-6m; SDK2e-4m; statefields1e-12m.
All184320new trainown/shuffledoraclecoefficients receive convex certificates
gap<1e-7m using eps1e-9m; unchanged heldcapacityaudit inherited.
ALLheldNNcoefficients/scores independent NumPy forward max<2e-4; winners and
decodedflows rebuild1e-12m, everyflow remains in its auditedsegmentfamily
(learnederror cannot beatoracle by>1e-7m). Allparent/subgroup/baseline/gates
recompute<1e-9mm. All3initializations/schedules/optimizer step1500 match;
first3updates each independentlyNumPygradient/AdamW replayed (9total),
parammax5e-5/lossmax2e-5. Remaining4491updates NOTindependently replayed.

Fixed commit/card/protectedsource/URDF/oldrun SHA,uniqueoutput. OneGPU,
<=900s/512MiBnewrun, within300GB total/3Oct23:59Beijing. Stop OWNchild only on
budget/input drift/contention/nonfinite/failedaudit. Preserve failures and
successfulfits, no restart on polling timeout. Archive allactual artifacts.
Viewed same-seed data are a diagnostic, not formalValidation/generalization.
Specificnovelty, causalrandomizedcommand robustness and learnedpolicyutility
remain unestablished even if this screen passes. GoalACTIVE,journalNOTREADY.
