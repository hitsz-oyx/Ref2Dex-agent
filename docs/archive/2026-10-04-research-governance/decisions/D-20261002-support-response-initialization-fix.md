# Engineering correction before physical collection

Runr1code5140208FAILED35.845s at initial XY staging check, before any controlled
physical trajectory or model fit. No label or scientific gate is available.
Preserve r1scripts, manifest and log. The collection requires correct GPU state
initialization, not relaxed pose tolerance or discarded environments.

Installed NVIDIA SDK primary documentation:
`/home2/wyy/isaac-gym/isaacgym/docs/programming/tensors.html`, Limitations,
states that refresh after setter without simulate can return stale data; setters
must be combined per step. Official NVIDIA repository also says refresh once
per step before setters:
[Factory environment source](https://github.com/isaac-sim/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/factory/factory_env_insertion.py).
Native reset itself calls indexed root setters for multiple actor sets and then
refreshes, making an extra perturb-and-refresh inappropriate.

Choose a reset transaction in a NEW v2collector: collect native reset setter
requests, suppress their immediate refreshes, retain native-written reset
buffers, apply the SAME initial XY offsets, then commit the union of root actors
once and union of DOF actors once before firstsimulate. No added settling tick,
no new q/state recipe, no placement range/seed/action/timing/label/gate change.
The saved initial state is the committed reset packet, not asserted to be a
refreshed solver observation before simulate. Save actor-index union and events;
verify SDK actor IDs. Allpostinit root/DOF state setters and external force calls
remain forbidden. First physics tick is counted normally. Native observations
computed during reset are not used by our actor; its currentq/object cache input
is built explicitly. Current model features are queried after>=36real ticks.

No old baseline or r1artifact modified. Unique r2output, same1800s/2GiB limit.
This fixes engineering execution; science conclusion remains pending. If queue
commit fails, stop and retain diagnostics instead of changing scientific gates.
