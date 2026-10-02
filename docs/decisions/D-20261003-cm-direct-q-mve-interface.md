# Decision Memo: direct-Q Cm MVE interface

Date: 2026-10-03
Decision type: bounded decision-interface Probe

## Question

Does the proposed decision-time short rollout become useful when Cm supplies only
object consequences and the continuation is the frozen direct-Q baseline?

## Evidence

The held 209,788-transition audit used the same physical checkpoint and current-state
projection. Direct-Q continuation produced object-projected MVE RMSE 26.60, row
Spearman 0.591, and episode Spearman 0.741, versus direct-Q's 26.83, 0.589, and
0.743. This is a small target-quality diagnostic shift without an episode-level gain.

The fresh native five-arm panel yielded 119 complete windows on 86 environment
clusters. Object-MVE changed 29 actions. Against Cup, cluster-bootstrap lower90
bounds were `-34.02 mm` for last-three-step minimum height, `-0.173` local reward,
`-0.215` contact fraction, and `-0.251` clearance. Its random-panel height/reward
Spearman was `0.360/0.503`, below direct-Q's `0.417/0.531`; random support was only
22 rows.

## Decision

Close this exact direct-Q MVE action interface. The model's object prediction still
contains an offline critic-quality clue, but it does not produce positive native
candidate utility or stronger action ranking. Do not train PPO, lower support or
utility gates, or expand ordinary Cm data under this contract.

## Cost and safety

The native run used one idle GPU, 96 environments, the frozen checkpoint, and an
82-second process budget. No external project, checkpoint, or other GPU process was
modified.
