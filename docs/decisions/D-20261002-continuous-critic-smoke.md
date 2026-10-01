# Engineering-only continuous target/critic integration

Follow the continuous-critic decision without claiming policy utility. One fresh
native panel567,768env,202ticks, same base/reference/XY/physics as prior task.
Four balanced preassigned groups64/motion: unchanged base, untrained continuous
actor+Cm critic, same actor+state-only auxiliary, same actor+no-aux critic.
All heads initialize seed762with zero actor/value outputs and Gaussian request
std.05; NOoptimizer update, no RL scientific gate or checkpoint selection.

Private group seed567+16000; three private CUDA request generators use567+17000,
giving common standardized initial random requests across groups, not paired
physical environments. Normalize causal executed target-minus-currentq12by
absolute native independent PDscale. Gaussian likelihood over requests, not
clipped many-to-one targets. Save all current raw states, requests/mean/logp,
executed targets/action12, state-only value, auxiliary predictions and true
one-step physical targets. Preserve original physical105as an audited label,
never infer utility from this wiring panel.

All actual model computation/native simulation uses one freshly admitted GPU.
Tiny deterministic initialization CPU to avoid startup; independent CPU NumPy
forward/probability/geometry/PD replay deliberately audits the GPU path. Whole
<=900s/512MiB. Hash source/base/references/own head packet before/after. Full
independent current-state/projection/likelihood/critic/next-target audits required.
Initial official evaluator checkpoint only bootstrap, never acts/initializes heads.
No external forces, state writes after reset, physics/material changes or push.

Successful smoke proves integration ONLY. Subsequent actual20-panel training and
fresh final evaluation need their own frozen runner/card/code and prospective
labels. Generic auxiliary PPO remains established, journal target unproved.
