# P-20260924-object-specialist-router

- Classification: Decision Probe.
- Cm: off. Every expert checkpoint is self-trained; no official actor.
- Route: fixed object identity from simulator metadata, hence a privileged
  diagnostic baseline, not an observation-driven deployable policy.

## Question and decision

Can a fixed portfolio of already trained self-trained actors demonstrate
simultaneous successful grasp on several corrected GRAB trajectories where
one shared actor scored 16/128? Freeze the route using prior seeds201/202
and duck seeds206–208: airplane e260, duck specialist e340, mug train5
e320, toothpaste balanced e360, alarmclock mixed12 e300; other objects use
e260. Evaluate first full episodes on entirely new seeds211–213, 64
environments each, with exactly the same 12 motions and disabled early
termination. Compare pooled held-lift with the previous best fixed actor
(21/128 on seeds201/202) only as exploratory context because the seeds
differ. Report every object's denominator, contact and failure count; do
not infer broad GRAB coverage from a pooled win.

Pass the *coverage* gate only if at least four object identities each
achieve >=25% held-lift across seeds211–213 and pooled success >=25%.
If passed, specialists are a viable multi-trajectory substrate; move to
observation-driven routing and then matched Cm value tests. If not, add or
repair object specialists before a Cm router. One or two idle GPUs,
<=30 minutes, <200 MB output. Stop on checkpoint/config/data drift,
GPU conflict, inconsistent route or incomplete evaluation.

## Results

Pending.
