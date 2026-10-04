# Diagnose the first native allocator failure; preserve the incomplete comparison

Actual resume-r1completed u02, native549and its independent panel audit, u03
and both first-minibatch gradient/Adam audits. It then FAILED at native550after
21.713s. First signal: PhysX allocator cannot allocate67108864bytes, result2;
laterCUDA700illegal address errors are not independent failure mechanisms.
Only initial.pt/physical_metadata/run_manifest exist for550; no trace/results,
u04or final568/569. Partial training remains INCOMPLETE, no utility label.

Known executable failing feedback loop: exact recorded native550command,
checkpointu03SHAac3d702bd0086f848e4cf4ed7c92605a7620b9a39733c525372500f9cb23d202,
GPU1UUIDfixed. It has already failed in21.713s. Native data/seed/model/task
remain fixed; no later model or alternate trajectory may be selected.

Ranked diagnostic hypotheses: (1) native scene/contact allocation exhausts
GPU memory at first simulation; prediction: low driver-free memory before the
allocation, even if model tensors are small. (2) unused PyTorch reserved memory
starves PhysX; prediction: reserved-minus-allocated is substantial at failure.
(3) device mapping/external contention: prediction: own process uses a GPU other
than admitted UUID1or driver-free memory drops without increased own allocation.
(4) invalid learned/native inputs: prediction: nonfinite pre-step goal/q/root.
Current logs do not choose among these. Unknown GPU0/2/3processes untouched.

Cheapest next action is ONE<=120sengineering-only reproduction of the EXACT
550/u03/native configuration, with passive per-step memory/finite-input probes
and own-PID GPU observations. This is never a training/evaluation panel and
never consumes a new optimization update. A foreign-GPU observation aborts only
the new owned process. No physics, environment count, action, model, reward,
seed, tolerance or allocator configuration changes in this reproduction.

Native graphics is already disabled by the headless task; do not assume its
graphics CLIargument caused the failure. Source failure and every partial
checkpoint/audit remain immutable. Original reserved400splus resume167.819s
plus diagnosis<=120s remains within3600s; include all own retained bytes in6GiB.
After diagnosis, choose an evidence-supported runtime repair or close as an
execution/implementation failure. Do not upgrade incomplete cohorts to a result
or launch a parameter scan. The journal objective remains active and unproved.
