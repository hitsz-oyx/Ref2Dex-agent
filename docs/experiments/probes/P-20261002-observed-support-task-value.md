# Current observed support information for eventual task value

Decision Probe. Fresh FIT597 and TEST598,768 environments/202 ticks each,
fixed stochastic u00 P0+zero-mean sigma.05 one-tick requests; all three request
actor states identical and updates0. source_e260 is construction-only.
Old547/595/578--586 and engineering panels excluded from fitting/evaluation.
Native PD, private placement/assignment, gravity, synthetic reference plateau
and physical105 criterion unchanged. No policy update, teacher action or Cm fit.

Only PRE ticks2<=t<201 with learned-arm group>0 and task outcome still unknown:
failure iff an ALREADY observed criterion tick violated the fixed physical
thresholds; success iff the complete105 window has finished without violation.
No future-success/contact/force selection. Target actual terminal105 bool,
gamma1 under the SAME fixed policy. Compact72=current normalized70 + alive
indicator + remaining known task ticks/202. Full support adds80 observations
computed solely from POST[t-1] and POST[t-2]: body origins15 and rotation first
two columns30 in the current object frame, object/five SDK body net force
vectors18 in that frame divided by actual object weight, prior root-height and
complete-mesh clearance increments2 divided by.005m, previous difference in
object-frame body positions15. These are measured SDK origins and net forces,
not true contact points/pairwise forces/force-closure certification. No FK
frame fitting, post-action force or future physical successor enters a model.

Two matched152->64ReLU->64ReLU->1sigmoid networks: support_value takes all152;
compact_value takes compact72 and zeros80. Common init3541, each1500 Adam
updates3e-4/batch4096/same private CPU batch stream3542 transferred to GPU,
clip norm10, binary crossentropy. Support extra mean/std from FIT only, floor
.001, clamp10. Compact normalization remains the frozen P0 normalization.
No early stopping/model selection. Third strong control: FIT-only motion x16
time-bin success mean (t*16//202 clipped15), empty cells global FIT mean.

Primary pooled held-out Brier on ALL eligible TEST rows/all three motions and
learned arms. PROMISING iff support error <=.99 BOTH compact and motion/time
and upper paired exploratory95%CI(support-control)<0 BOTH.2000 bootstraps3543
over192 common-noise episode groups retaining their three replicas and row
weights. FIT>=10000/TEST>=1024/episodes>=32 else UNCLEAR without alternate
data. No subgroup rescue, threshold changes, width/update/seed/target scans.
Positive demonstrates useful CURRENT observed information only, not Cm
action information, policy benefit, novelty or formal Validation.

Native independent full raw70/context, P0/request/critic/PD/geometry/105 audit
on BOTH cohorts. Separate audit rebuilds raw compact features/task history,
actual SDK transforms using Torch64 quaternion matrices, previous net forces
and flow, FIT statistics, ALL TEST network forwards via NumPy <=2e-5,
controls/cluster statistics/gates. No optimizer replay claimed. SDK rounding
reconstruction absolute limit2e-5; FIT statistics exact on same raw float32
decoded observations. GPU preferred for physics/model computation, CPU for
pure reconstruction/statistics. Whole<=600s/768MiB, original inputs pinned,
owned-child-only termination, no checkpoint/output overwrite.
