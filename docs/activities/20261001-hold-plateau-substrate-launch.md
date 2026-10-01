# Frozen hold-plateau substrate launch

The single-session implementation adds an actor-only native collector, a bounded
four-panel runner, and an independent NumPy trajectory scorer. The original
workspace, reference tensors and checkpoints remain read-only. Only synthetic
references prepared in run `P-20261001-hold-plateau-reference-r1` are loaded.

Six focused checks pass: 74/75-step boundary, phase-only followup, interruption
by either force proxy, own initial native height, discontinuous progress, and
incomplete plateau rejection. No model training or controller tuning is included.
All 148 inherited/protected inputs match before launch. Native rollout length
2000 exceeds all new reference lengths; first-episode endpoints should therefore
be 632/742/577 physical steps. The collector additionally checks the actual
loaded reference paths, lengths, plateau poses and seven native velocity tensors,
and verifies progress before and after every active first-episode step.

Run identity: `P-20261001-hold-plateau-substrate-r1`. Actor286/287, init498/499,
192 environments per panel, 64 per motion. GPU5 is admitted freshly per native
panel; its last read showed 4MiB and no compute process. Whole-run cap1800s,
storage256MiB. CPU is used only for saved trajectory label/statistics recomputation.
The frozen gate is pooled75-step phase retention >=10% and >=5% per motion.
Failure ends this task variant/actor pair; it does not justify metric changes.

Runtime output and manifest:
`src/task/CmResidual/research/contact_response/output/P-20261001-hold-plateau-substrate-r1`.
Scientific status remains pending until all768 first episodes and the independent
trajectory audit complete. This is synthetic-task feasibility, not Validation.
