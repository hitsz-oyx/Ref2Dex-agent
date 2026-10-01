# P-20261001-fresh-causal-transfer

Decision Probe. Frozen before fresh simulation; secondary nominal hypothesis
was motivated by reused historical data. No Validation or novelty claim.

## Question and next decision

Can the frozen nominal-motion effect models predict new plus/minus action
contrasts better than frozen command models and the zero-effect control?
Positive results justify action ranking on a new bank; failure terminates this
direct factual-predictor route. Do not refit after inspecting new outcomes.

## Fixed design

- New reference panels: existing plain actors286/287, new environment seeds
  490/491. Same one airplane/three motions,96environments balanced32each.
  Cold simulator per reference and each intervention. Actors/checkpoints are
  read only. Fresh initial randomness is not new object or actor diversity.
- Compact reference up to360steps, stopping at the first native done event.
  Reference actions and CPU/CUDA RNG are captured, full initial tensors/player
  state/properties restored and verified. Actors and RMS stay frozen.
- Reference-only acquisition: first pre-action step>=10, at least5subsequent
  valid steps, both current independent force proxies true, and full sampled
 1538hand/1024object minimum gap<=20mm (seed42 assets). Scan geometry only
  on eligible force states. No future object outcome selects a trigger.
- No-trigger environments remain untreated and are explicitly excluded. Require
  >=256valid windows total and>=48in every96-environment panel; otherwise
  UNCLEAR acquisition and do not continue physical arms. Report every motion's
  coverage, including zero. Do not tune thresholds or substitute seeds.
- Arms zero_a,zero_b,x_plus,x_minus,y_plus,y_minus,z_plus,z_minus: one normalized
  native wrist-translation pulse of+/-0.01in the named WORLD axis, exactly one
  step. Native translation PD scale1 must be verified for all axes. Reference
  actions/RNG follow thereafter. No clipping permitted. No other action changes.
- Zero arms measure repeatability. Every paired arm must match the actual
  pre-intervention object positions to<=0.05mm RMS; max per-coordinate joint
  mismatch<=1e-4and object orientation mismatch<=1e-4. Larger mismatch is invalid
  implementation; no post-hoc removal. Hidden solver identity is not guaranteed.
- Frozen effect checkpoints from factorization r1: nominal_motion,raw_command,
  state_only, seeds411–413. Hashes fixed before fresh simulation. Current-state
  features/exact nativePD+FK identical to fitting. Predict one-step world object
  displacement: current velocity/30 plus learned local residual rotated by the
  CURRENT object rotation. Future object/hand state enters measured outcomes
  only, never available model inputs. No training or model selection here.
- Primary target: world one-step displacement difference, plus minus minus,
  at the identical baseline schedule. Report vector contrast RMSE against the
  measured physical difference, vector cosine, magnitude, all axes/panels/motions.
  State-only/passive controls predict zero contrast at identical state; actual
  replay-state differences are retained and their contribution reported.
  Five-step contrasts are measurement diagnostics, not one-step model scores.

## Frozen gate and limits

PROMISING requires pooled one-step pulse RMS>=2times zero-repeat RMS (if repeat
is0and pulse>0, observed-noise comparison passes without inventing a finite ratio),
nominal seed-mean contrast RMSE>=10%below EACH raw-command and zero-effect RMSE,
mean cosine>=0.5on fixed nonzero-target contrasts (norm>1e-6m), and nonnegative
RMSE gain versus raw in every optimization seed. Else UNPROMISING. No change
to earlier labels; no selecting a favorable axis, panel, seed or horizon.

One idle GPU4,2CPUthreads,<=3600seconds total,<=4GiB. Sequential processes.
Reference and arm sources/input SHA verified. Failure preserves artifacts and
does not automatically restart. No videos, PPO training, external writes or
original-worktree mutation. Contact proxies+sampled geometry are not proof of
pairwise force attribution; test intervals are scheduled from reference states.
