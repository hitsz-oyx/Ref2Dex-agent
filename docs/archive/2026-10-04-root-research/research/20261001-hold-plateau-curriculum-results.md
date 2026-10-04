# One frozen holding curriculum continuation failed

Code `e770d4b`, run `P-20261001-hold-plateau-curriculum-r1`: COMPLETED in1120.95s,
106.30MiB on GPU5, all owned processes exited. Training939.63s, exactly300 NEW
epochs/460800 interactions. Fixed seed721, self-trained actor286e420 source,
fresh optimizer, predeclared reset/reward/exploration recipe. Final checkpoint
SHA `73112ee2fc9aa103e3c0be742803a4eb9d206dedb1f1aac405eb01c74c5861f2`.
No intermediate/best checkpoint or additional seed was selected.

Source network and both RMS fingerprints independently match; all final weights
are finite, wrist exploration remains5mm and other sigma entries unchanged.
Checkpoint reset counts by motion (frame0/plateau) are184/77,165/59,215/74.
The68947 training plateau samples and7933 positive dense-bonus samples are
privileged training diagnostics and never evaluation success.

Final-only tests on500/501 completed384 first episodes,128per motion. Native
plateau/progress/endpoint checks pass, and the evaluated model/RMS fingerprints
exactly match the final checkpoint. Independent saved-trajectory audit agrees
with every online count. Retained75=0/384, stable45=0/384, every motion0/128;
mean phase root rise0.269mm and conjunction of force proxies0 throughout phase.
All four feasibility gates fail, UNPROMISING. All178 protected input hashes
remain unchanged. Stop this specific PPO continuation, without duration, rate,
seed or sigma tuning. This does not establish that every curriculum is ineffective.

Three-pose native reset label audit (`da39660`,0.739sCPU) applies the ACTUAL
`Dexplore_Inspire._set_env_state` method to dummy tensors. Maximum joint changes
relative to raw reference are0.29677/0.26261/0.09957rad. Sampled minimum surface
gaps change0.9135->0.3453,0.9456->1.4025,0.4883->0.8457mm; existing geometric
finger-contact flags remain3/1/1. Thus the raw poses are changed by native
coupling/clamps, but this audit does NOT support blaming loss of minimum
proximity or changed binary contact flags for PPO failure. Unsigned geometry
does not establish frictional gripping, nonpenetration or attributed forces.

Next cheapest distinct question: can exact static PD targets preserve these
elevated, zero-velocity native poses at all? A reference-pose mechanical probe
can distinguish an available unmodified static controller from another failed
learner without more PPO. It must additionally save full object pose and enforce
conservative object-table clearance; it is neither frame0 pickup nor a learned
actor, and no success in that diagnostic can replace the failed baseline gate.
