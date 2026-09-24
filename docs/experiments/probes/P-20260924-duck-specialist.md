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

Pending.
