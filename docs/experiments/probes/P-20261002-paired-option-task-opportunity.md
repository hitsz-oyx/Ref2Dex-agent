# Paired executed joint-option task opportunity

Decision Probe, fresh601/768env/202ticks,3motions x64groups x4arms. No training,
official/source actor actions or failed Cm guidance. Self-trained P0 unchanged.
Groups0/1 P0 duplicates,2 fixed option,3 antithetic option. Original balanced
private arm permutation seed601+16000. For each motion/group, one private XY
placement draw uniform +/-1cm (192x2 CPU generator601+11000) is copied to all
four arms at INITIAL reset only. Group identity within each motion/arm is
sorted environment order. This is new matched data, not reusing old unpaired
duplicate counts as repeat variability.

Joint options draw192x12 CPU standard Gaussian seed601+18000 exactly once;
group2 raw=request,group3 raw=-request. Same native executable-target decoder
and fixed scales .02mXYZ/.10radrotation/.15rad independent fingers, tanh,
native bounds/couplings. All arms common P0 BEFORE PREtick lift_start-8.
Options held from that known PREtick through the known physical105 deadline
stop+30 exclusive in PREclock; thereafter P0 again. Current P0 feedback/PD
recomputed every tick. No object/state/force write after initial reset, no
future observation triggering, no repeated random draws or dose/timing scan.
u00 checkpoint is construction metadata only; its request actors/critics do
not act. Their unused probability/noise/value channels are explicitly zero,
not a claim that these interventions follow the old sigma.05 policy.

Engineering gates: identical initial per-group object roots/nativeq/velocities
within1e-7; on ALL common-prefix PREticks including decision PREstate, per-group
q XYZ/objectXYZ discrepancy<=.5mm, q rotation/fingers<=.005rad, translational
velocities<=.01m/s, angular/joint velocities<=.1rad/s, object quaternion
rotation matrix entry discrepancy<=.005. These are prospective matching
limits, not an assertion of exact simulator counterfactuals or full hidden
contact-solver state restoration. Both P0 physical105 labels agree>=95% of
192groups. Failure gives UNCLEAR with conditional interpretation disallowed;
do not widen matching limits or selectively retain matching groups.

After all engineering gates pass, primary opportunity ceiling is whether the
retrospective OR of anchor P0arm0 and option2/3 outcomes improves pooled physical105 by at least
5 percentage points over BOTH P0 duplicates and improves each motion by>=0
against BOTH. At least5groups must have different option2/3 outcomes. ALL192
groups included; no subgroup rescue or select failed-only groups as primary.
Success label PROMISING for executed opportunity ONLY. Oracle uses future
outcomes and is not deployable; random candidate mean/worst and all arm/motion
counts must be reported. No model/selector/learning utility or formal Validation.

Full native reset/private randomization/actual initial cloning, P0 NumPy forward,
causal70 context, fixed option draw/start/stop, exact target projection/PD,
full-mesh105 labels and actual matching metrics independently reconstructed.
Known SDK body origins/net forces logged for later mechanism designs, no
claim of contact points. Original native numerical limits retained. Whole
<=600s/512MiB, freshly idleGPU6/4/5/0/7, CPU pure audit/statistics, input hashes
guarded, no overwrite or unknown process termination. Positive needs a new
frozen learning design; negative ends this family without additional candidates,
amplitude/axis/duration/seed/checkpoint scans.

Pre-data design review: include the unchanged P0arm0 as an actual candidate
fallback. A selector can reject both intervention options; requiring options
alone to dominate P0 would ask a different question. No601data/model/physics
exists at this revision; all matching/effect thresholds otherwise unchanged.
The duplicate P0arm1 is a repeat control, not another oracle candidate.
