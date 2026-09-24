# P-20260924-duck-specialist

- Classification: Decision Probe.
- Cm: off.
- Source: self-trained s3 airplane e260, SHA256
  `16fd261b4b2de4cbdb257b09f1c7b363b384153103901ff831c825cf47d6a78f`.

## Question and decision

The official actor diagnostic on corrected duck was 64/64, whereas the
12-motion shared self-trained actor achieved 0/12 on duck across seeds201/202.
Can 80 epochs of duck-only continuation from the same self-trained e260
source acquire duck held-lift? If fresh seed206 yields >=16/64 successes
and contact fraction >=0.30, retain object specialists as a practical
multi-trajectory baseline route; then check a second hard object before
building a router. If it fails, prioritize reward/curriculum or starting
representation, rather than another broad mixed continuation.

Use fixed corrected duck tensor, 64 environments, seed70, e260→e340,
the same 2/10/5 approach/held-lift/progress reward and contact curriculum,
one idle GPU, <=60 minutes and <5 GB outputs. Stop on source/input drift,
GPU conflict, nonfinite training or missing e340 checkpoint. Evaluate 64
first full episodes on unseen seed206. This is a Probe, not a stable
multi-seed claim.

## Results

Training `agent_duck_specialist_s70_e340` completed on commit `a9acc7c`.
On unseen seed206, the duck specialist achieved **48/64 held-lifts (75.0%)**,
mean hand/object contact 0.590 and mean maximum contact-supported lift
0.308 m. The frozen e260 source on the exact same duck motion and seed
achieved **1/64 (1.6%)**, contact 0.166 and maximum contact-supported lift
0.005 m. Both prespecified duck gates passed. This is a single training
seed and first eval seed, hence `PROMISING`, not a formal claim that all
duck starts are solved. Repeat on seeds207/208 before choosing a reusable
specialist portfolio. The two additional unseen evaluation seeds scored
**49/64** (seed207) and **43/64** (seed208), totaling **140/192 (72.9%)**
across seeds206–208. This replication strengthens the Probe signal without
turning it into matched multi-seed training Validation.

The duck result changes the route: shared-policy interference or limited
per-object update budget is a plausible main blocker, while the fixed reward
and curriculum can in fact learn this object. Next test the same specialist
protocol on waterbottle, another physically feasible identity with zero
shared-policy held-lifts. Waterbottle-only continuation from a *different*
train5 e320 source previously collapsed; the new e260 source tests whether
that failure was inherited from the mixed checkpoint. No hyperparameter is
chosen from duck's heldout seed.
