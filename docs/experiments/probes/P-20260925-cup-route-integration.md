# P-20260925-cup-route-integration

- Classification: Decision Probe.
- Cm: off. All six experts are self-trained PPO checkpoints.
- Route remains a simulator-object-ID diagnostic, not deployable
  observation routing.

## Question and decision

Does the cup e340 specialist's 180/192 single-trajectory held-lift
signal survive in the mixed 12-motion simulator and raise pooled
coverage? Freeze two routes: previous five-expert route and the same
route with only `cup` changed to cup e340. Evaluate both on entirely
new seeds229–231, 64 first full episodes per seed, disabled early
termination. Use the same motion pool, checkpoints, environment config
and strict held-lift criterion; record per-object denominators.

Pass if new route improves pooled held-lift by >=9/192, cup contributes
>=12/18 held-lifts, and non-cup successes fall by no more than 6/174.
If passed, update the observation-driven router for the expanded expert
portfolio and then test it online. If not, diagnose whether the cup
specialist is sensitive to mixed-environment conditions. One idle GPU,
<=30 minutes, <200 MB outputs. Stop on checkpoint/config drift,
GPU conflict or incomplete episodes. This is a Probe, not formal
matched multi-seed training Validation.

## Results

All six runs completed on GPU 7 at commit `334f6ca`, with identical
motion IDs and start frames for all 64 environments within each seed.
Run manifests and episode results are under
`outputs/CmResidual/agent_router_{old,cup}_s{229,230,231}/`.

| Seed | Old total | Cup route total | Old cup | New cup | Old non-cup | New non-cup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 229 | 21/64 | 24/64 | 0/6 | 6/6 | 21/58 | 18/58 |
| 230 | 20/64 | 27/64 | 0/6 | 6/6 | 20/58 | 21/58 |
| 231 | 16/64 | 24/64 | 0/6 | 5/6 | 16/58 | 19/58 |
| Pooled | **57/192** | **75/192** | **0/18** | **17/18** | **57/174** | **58/174** |

Per-object pooled old → cup-route held-lifts: airplane 14→15/45,
alarmclock 1→4/18, apple 2→1/15, cubesmall 1→1/18, cup 0→17/18,
duck 14→15/18, mug 12→10/15, phone 0→0/15,
toothpaste 13→12/15, waterbottle 0→0/15.

**PROMISING:** all three predeclared gates pass: +18/192 pooled,
17/18 cup and +1/174 non-cup. This establishes mixed-environment
headroom for the cup specialist under the privileged object-ID route.
Next check whether an initial policy-observation classifier can select
all six experts on a held-out seed and then preserve the online benefit.
These are single-training-seed Probe results, not a generalization
Validation or Cm policy-utility evidence.
