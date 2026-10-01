# HF15 slot2：真实完整程序后果与闭环选择

Decision, current session. Candidate opportunity is PROMISING: rotation_cup7
vsbase+26.509mm signed retention, both held90% intervals positive, original
risk/support gates pass. This is candidate gain, not Cm utility. Fit-fixed7
is mandatory comparator; selecting7everywhere cannot establish Cm gain.

Train only new actual completeH10 records from seeds461–472. Same frozen
initial-motion/start split9851: fit1656/cal586/held1675. No old2+8Cm weights or
boolean contact backfill. Inputs pre-action69D×10history, native observation
at trigger ONLY (future native observations are audit data), candidate first
12independent commands, fixed8program identity and rotation anchor3. Program
identity represents the full feedback law, not a first-command-only forecast.

Physical32targets: signed10height changes/.01m,10source-plane-clearance
changes/.01m,10joint normalized-net-force presence bits, last3joint+clearance
retention and any10step clearance-loss. History/native fit-only mean/std
floor.001 and clip8; anchor scalesπ, no future physical input. GRU64 +native
MLP64, base head128, relative-to-base program effect128. Strong state-only
GRU/context with eight static catalog heads gets NO current action/anchor
tensor; head identity still encodes fixed program, explicitly reported.
Shuffled uses identical Cm topology and permutes complete program descriptors
within fit pre-action clear/nonclear strata, leaves labels unchanged.

Three matched initialization/minibatch seeds10501/10502/10503 permodel family,
1000Adam updates each, batch128,lr.0003,weight_decay.0001,gradclip1; fixed budget,
no early best-checkpoint or held tuning. Common physical Huber20coords+BCE12
bits. Full actual batch sampling same perseed; scalar future value/V unused.
Record held factual heightRMSE/clearanceMAE, joint Brier and predictive benefit
against both controls, separately on initiallyclear. Honest predictions over
all8programs permit cal-only policy preparation, no individual oracle/regret.

Cal-only nonnegative-slope affine logit calibration of retention/release,
proper BCE,200updates/.03, ensemble means; save source hashes and coefficients.
Require >=5positive/negative calibration samples on initiallyclear for each
event; insufficient makes utility setupUNCLEAR, no probability claims. Per
mode intervention margin=max(2mm,75thpercentile absolute factual-score residual
on calinitiallyclear). Common score uses minimum predictedlast3positiveheight
above rest times calibratedretention probability minus startingpositiveheight.
Candidate mean advantage must exceed margin+ensemble score-difference std;
release upper ensemble bound<=base+2pp and retention mean>=base−5pp, otherwise
base. InputOOD unclippedabs>8 also abstains. Freeze all learned weights/
calibrations/margins before independent direct-control data. No utility based
only on retrospective matched factual rows or model loss.

Independent randomized fresh five-recommender probe is required: Cm, strong
state-only, shuffled, base and frozenfixed7, with merged actual program
propensities, actual execution/reobserve/control coverage. Planning H10 but
executing2/reobserve is a new closed-loop policy, not the factualH10 endpoint;
record its actual local trajectory and task utility separately before PPO.
Freeze this fresh collection/gates in its card before launch. For future
primary mechanism gain require enough actual distinct recommendations and
matched labels, positive score vsbase AND fitfixed and both learned controls,
risk bounds and two cluster90%lower bounds>0; support missingUNCLEAR. No
success-rate/PPO sweep or stable-grasp claim at this stage.

Slot2<=60min/8GiB total fit+calibration+engineering+freshutility. Fitter gets
<=600sec/1GiB, reserving remainingtime for actual utility. GPU0/1 fresh admission,
own unique output/no old writes, source/data/actor hashes pinned, process
terminal/input audit. NN prototype GPU smokes synthetic only,1.913sec, no
actual-data fitting/utility and no saved learned checkpoint. Mission/C3 unchanged.
