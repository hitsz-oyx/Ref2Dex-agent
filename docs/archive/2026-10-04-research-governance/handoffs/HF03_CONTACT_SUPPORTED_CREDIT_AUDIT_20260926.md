# HF03 contact-supported credit CPU audit handoff

This is a CPU-only preflight for the deferred contact-supported credit idea. It is not an HF03 online experiment and did not use GPU, PPO, collector, or Isaac Gym.

The audit fit on seeds 246–247 and held out seeds 248–249 from the existing route-specific transition substrate. The actor checkpoint SHA256 is `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`. The audit has 123 fit rows and 122 holdout rows, with balanced plus/minus arms. Its scope is deliberately narrow: one recorded `motion_id=0` on the airplane/source-e260 substrate, so it makes no cross-object claim.

The predeclared gate required the post-handflow model to improve both held-lift Brier and maximum contact-lift RMSE by at least 5% over the action-aware and post-handflow-shuffled controls. On the holdout:

| model | held-lift Brier | max contact-lift RMSE |
|---|---:|---:|
| action-aware | 0.18676 | 293.12 mm |
| post-handflow | 0.19111 | 287.66 mm |
| post-handflow shuffled | 0.18313 | 302.34 mm |
| state-only | 0.18699 | 284.63 mm |

The gate failed: post-handflow is worse on Brier than action-aware, and its lift-RMSE improvement is only 1.86%, while the shuffled control is better on Brier. The audit is therefore `UNPROMISING` for promoting this representation to a PPO auxiliary Probe. It does not close the broader research debt because the substrate is one motion; any future HF03 work would require a new cross-object data contract and experiment ID.

Reproducibility artifacts:

- report: `outputs/CmResidual/agent_contact_supported_credit_audit_20260926_r3/report.json`, SHA256 `265a06272f460ff2ec48513496c42b990e9ff9c9c87dd5ed97fd70e8d89a52e3`;
- manifest: `outputs/CmResidual/agent_contact_supported_credit_audit_20260926_r3/run_manifest.json`, SHA256 `93c92ea2c5d425efe6fa0045b7afb45c2acff4bc67fc52bd39cd09ce9972b888`;
- fitted CPU models: `outputs/CmResidual/agent_contact_supported_credit_audit_20260926_r3/cpu_models.pt`, SHA256 `3838f29ca827e42736ca7e68b6fb05c4c07ae889227b420f18bdc18ca6e9ebb5`;
- source script SHA256: `f71c09edc4868eed8e269a99692c2758af7f1ef09abf7ec47773c37e47cfe42f`.
