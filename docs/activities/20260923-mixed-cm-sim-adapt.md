# 2026-09-23 mixed Cm sim-adaptation operation

Operation: CPU-only small-data adaptation Probe of ObjectInteractionCm V1.3
mixed GRAB MANO + Inspire initialization versus same-architecture scratch.

Run ID: `agent_mixed_cm_sim_adapt_40step`

Code commit: `7615458755b0538819d6520916280dbee128639a`

Input protection: pretrained and four transition files checked by SHA-256;
external Ref2Dex project remained read-only. Train seeds 74/78 and evaluation
seeds 95/96 were fixed in the committed script. GPU use: 0; CPU threads: 2.

Run status: COMPLETED. Started 2026-09-23 15:32:10 UTC, ended 15:33:17 UTC.
Final step: 40 per arm. No checkpoints were saved; this was a decision Probe.
Output: `outputs/CmResidual/agent_mixed_cm_sim_adapt_40step/report.json` and
`run_manifest.json`; output size 16 KB. No process remains running.

Scientific interpretation is in
`docs/experiments/probes/P-20260923-mixed-cm-sim-adapt.md`.
