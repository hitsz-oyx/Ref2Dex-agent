# Repair conditional oracle qualification after r1

Decision: how to test the user's oracle question when mixed actions change
same-env GPU PhysX futures despite a bit-identical prefix?

Evidence: r1 completed1132.675s/5.198GB,6000Q updates/1920trajectories. All
actual labels/PD/contact/NN audits pass; queried/deployed futures fail. First
state difference follows1.86e-9N force difference while commands are identical.
No cross-env contact. Persistent host PD target engineering control reproduces
the exact same mismatch, excluding lifetime as the supported repair. Failed
engineering r1 input-path error is preserved; r2 inherits its completed baseline
under SHA and completes the actual retention test with no new fit.

Action: qualify CPU PhysX mixed isolation with CUDA P0 inference, same scene/
seed/options/timing, retaining the full PD target tensor without changing
decoding. GPU remains default for model computation. CPU physics here addresses
an observed conditional-query correctness limitation; it is not a cost shortcut.

If all short states/commands/forces match exactly, run r2 under CPU physics:
fresh TRAIN/EVAL/4fits/4deployments, same96envs/761/762/805/806/807,1500steps/arm,
decision36/H32/eightoptions/full105criterion and original5pp gates. No cached
GPU-physics labels/models reused. Archive r1 separately; never pool backends.
One GPU, <=3600s/12GiB new probe; CPU qualification <=600s/1GiB.

If CPU does not qualify, stop this mixed action-query implementation and use
a single controlled episode per otherwise fixed scene or current-contact
privileged teacher. Do not tune tolerances, horizons, options, seeds or call
the idea refuted. Full r2 query/deploy validation still required even if minimal
CPU engineering passes. Mission/claim unchanged; no external write boundary.
