# D-20260924: Reopen multi-trajectory and Cm Probes

The user's 2026-09-24 instruction supersedes the earlier Option A pause in
`D-20260924-after-task-aligned-option.md`. The user authorizes autonomous
baseline training, Cm Probes and simulation, and asks for choices to be logged
without another decision request before 2026-09-25 10:00 Asia/Shanghai.
The hard machine, storage, workspace and process boundaries in `CAMPAIGN.md`
still apply.

First choice: continue the self-trained s3 e260 actor on 12 already converted,
coordinate-corrected motions spanning nine object identities. Exclude
`s1_torussmall_lift` from the first training mixture because its official
diagnostic actor achieved only 6/64 held-lift; retain it as a stress test.
Use one idle GPU, 64 environments, identity-balanced motion listing, and a
fixed e300 endpoint. This is a cheap coverage Probe. It asks whether a single
actor can retain airplane grasp while acquiring several distinct trajectories.
The result determines whether to expand conversion/training, change sampling,
or first repair baseline coverage before a matched Cm-on/off policy test.

The 12 motions are a local converted subset, not the entire GRAB dataset.
The first baseline evaluation will report motion/object strata and held-lift,
contact, and failure patterns. Cm mechanism work will target contact-supported
future lift and require a matched same-source Cm-on/off/placebo comparison
before any utility claim.

## 2026-09-24 late-night resource choice

After the duck and waterbottle Probes, outside jobs occupied all GPUs.
Waterbottle start-reset annealing and the frozen object-specialist route
remain decision-relevant and are already specified in their experiment
cards. A bounded deferred runner will wait for a genuinely idle GPU until
2026-09-25 09:30 China time, use at most one GPU, and execute those two
fixed Probes sequentially. It stops on code/input drift, GPU conflict,
failure, or the wait deadline; it does not change scientific gates or
infer conclusions. This preserves the user's no-decision-request window
without interfering with other processes.
