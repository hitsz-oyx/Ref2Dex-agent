# P-20261001-contact-trigger-geometry

Decision audit, not policy or physical causal Validation.

Question: are the frozen pulse intervention states near the hand/object surfaces,
or do the independent force proxies mainly trigger without interacting geometry?
If >=80% of all384 zero_a states have sampled surface gap<=20mm, preserve the
schedule for a broader-coverage collection design. Otherwise stop using it as a
contact selector and select collection states using observable geometric proximity.
This threshold is a data-acquisition screen, not proof of pairwise physical contact.

Cheapest test: the existing384actual pre-intervention states; one forward-kinematics
and sampled surface-distance pass on an admitted GPU4, no simulation or training.
Use the pinned Inspire and airplane URDF, the existing deterministic1538/1024
surface sampler and fixed seed42, all sampled points with chunked cdist. Record
surface/asset hashes and gap quantiles for each panel/motion. Sampled minimum
distance is an approximation; it does not detect penetration or actual contacts.

Budget: one idle GPU4,2CPUthreads,<=120seconds,<=10MiB new outputs. Inputs read-only;
stop on missing geometry, nonfinite positions or changed input hashes. Reused
pilot data, three motions of one object, no generalization claim. Geometry is
computed only from current state; future object outcomes do not choose the states.
