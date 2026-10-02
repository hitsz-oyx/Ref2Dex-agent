# Measured geometry contract after failed FK binding

Native576 geometry logging completed; all original native physics/PD/request/full-mesh checks pass. The predeclared link-frame position binding fails: float32 maximum84.0425µm, same-data float64 maximum84.0360µm, limit5µm unchanged. Rotation2.381e-6 passes2e-5; fixed root exactly invariant. COM alternative28.07mm fails. Precision alone does not repair the contract; no unique solver/frame cause established.

Close historical q-to-exact-SDK-geometry reuse. Preserve both raw and failed FK outputs; derived_geometry.pt remains binding_valid=false and is not scientific measured training data. Future predictors use newly measured SDK body poses. Five body origins are not contact points, and no force closure or grasp value is inferred.

Engineering diagnosis summary correction: original per_body_mean_error_world_m was inadvertently a SUM, not mean. A separate summary-r1 divides by202*768. Max, RMS, raw physics, thresholds and false binding remain unchanged; original results untouched. The script is corrected for future use; no native/FK rerun.

Next Decision is task-aligned geometry: forecast root-height and full-mesh-clearance increments using actual current object-frame body poses and observed flow. Controls must isolate action, geometry, known phase and observed persistence. This is an investment screen toward DIRECT actor coupling, not journal novelty or policy utility. Closed COM-translation/impulse recipes remain closed.
