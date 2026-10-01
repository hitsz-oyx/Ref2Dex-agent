# 轻物体接触力代理的单位核对

Blocker engineering diagnostic, 2026-10-02, before any new plan-Cm labels.
Archived native properties from HF09 show airplane mass.002593613kg/flags0,
weight.0254433N vs current.1N threshold. Installed original SDK documentation
`/home2/wyy/isaac-gym/isaacgym/docs/_sources/programming/tensors.rst.txt`
lines279–281 says net-force values depend on contact collection/substeps and
all-substep forces are aggregated. Native config has4substeps. Mass alone does
not establish threshold error; the aggregation factor must be measured.

Known-static-contact positive condition:96env reference-frame0, zero initial
object velocity, hand translated2m upward and held by native PD. Preserve
asset/shape/density/gravity/contact settings and original collection mode.
Advance150native30Hz ticks. Final60frames classify quiet as linear speed<.005,
angular speed<.05 and hand-object distance>1m. Save raw object force, mass/flags,
gravity/mode/substeps, object pose/velocity, whole-mesh/table clearance and hand
force/distance. Report force/weight and old-threshold detections. No quiet
frames means diagnostic failed, no assumption of contact from a threshold.

This measures force scale on an externally supported quiet object, not grasp
success. No NN/Cm/V/PPO updates; no old labels/gates changed. HF13's geometric
loss remains evidence against pose-freeze even if force proxy proves unsuitable.
If calibration rejects the old threshold for known gentle contact, record the
measurement boundary and define a physical contact/clearance contract before
new labels; otherwise retain it and investigate active finger control.

Cost one admittedGPU/<=240sec/100MiB, native limit210sec, unique owned output,
96env owns only its new simulation. No external writes/process kills/identity/
claim/resource changes; no authorization boundary. First execute the reversible
calibration and archive exact results, do not infer force units from a mirror.
