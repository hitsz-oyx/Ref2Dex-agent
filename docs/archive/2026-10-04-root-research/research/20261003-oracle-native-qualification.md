# Attributed contact and fresh-scene replay qualification

Engineering only, not a grasp-utility result. Actual own native executions:
`engineering-oracle-contact-api-r1`, `engineering-oracle-native-replay-r1`
(FAILED device mismatch before physics control), and `...replay-r2` (COMPLETED).
No learned Cm/official actor/optimizer update. One admitted GPU6, bounded owned
process groups, CPU tensor reads + GPU PhysX + CUDA own-P0 inference.

GPU data pipeline contact API calls return an empty array while SDK logs say
unsupported after simulation starts. Python does not raise. The original smoke
field `attributed_contact_api_available` measured only absence of exception and
is misleading; `interpreted_capability_audit.json` preserves and corrects its
interpretation without rewriting original results. Net-force tensor alone
showed9.809994N while raw pair records were empty.

CPU pipeline + GPU physics returns16 box/table contact records across four
substeps and summed lambda9.809998N for a1kg resting cube. CPU physics likewise
passes. Lambda/weight~1, lambda/(weight*dt)~60: treat lambda as force-valued
under this collection contract, not as an impulse. Friction/tangent metadata
are all zero in both controlled and actual grasp recordings; these fields
remain unavailable/unqualified. No certified tangential-force or friction-cone
claim. Positive normal-force records are distinguished from potential contacts.

The local task-source snapshot fixes five implicit default-device placements
and twelve hardcoded CUDA reference transfers for the CPU tensor pipeline.
External task files are untouched. r1 retained the resulting device traceback;
r2 keeps identical physics, own scratch P0 and corrected shape ownership.

Full48-env scene/seed751,72controlticks,144physics frames: baseline and fresh
same-scene replay have identical initial packets, ALL absolute PD targets,
actions, all27-body states, native q/dq, object roots and net-force tensors.
Trace files are byte-identical (SHA3d97b666...). Fixed prospective prefix
tolerances were position/quaternion/joints1e-6 and velocities1e-5; all errors0.
No root/DOF write after frame0 and no cross-env cloning. Baseline36.15s and
replay19.30s including task initialization; original failure is separate.

998255 raw records include546549 table/object and many fingertip/intermediate
hand/object pairs;122968 records have positive normal force. Summing lambda
times normal, positive on body0 and negative on body1, reconstructs object's
net-force tensor with mean3.93e-8N/max1.418e-6N residual. This calibrates pair
sign and units for this scene/collection contract; it does not establish full
force closure. Contact positions come from substeps; final-frame pose alignment
is approximate, so object-local contact positions are used directly.

Decision: native interface and baseline full-prefix reproduction are qualified.
Proceed to the user-authorized no-Cm effect/interaction task-Q Probe. Candidate
prefixes and mixed selected futures must also be checked on actual new runs;
baseline replay alone does not certify those counterfactuals.
