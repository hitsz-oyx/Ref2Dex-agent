# Decision: can approach and lift reach the observed mechanical candidates?

Question: static suspended holding does not prove pickup. Can a fixed reference
tracking controller physically reach a holding state from original frame0?
Evidence: four-dose static gate failed;13/48 cases at.30rad retained75, motions0/2.
The.30rad candidate is selected POST-HOC from that failed screen, explicitly.
No further static dose tuning. Choose a distinct temporal reference controller,
fixed.30 versus0, with prospective fresh data. Cost<=600s/128MiB, one freeGPU.

At each pre-action progress p, use label wrist/finger q at min(p+1,plateau_stop).
Clamp native finger/coupling bounds, add closure ramp from15label frames before
first-lift interval to its start, remaining.30duringlift/plateau. Wristabsolute
reference unchanged; inversePD usescurrentrealizedq. No learned actor, action
clipping, object-state assignment after reset, teacher adaptation or refselection.
New seeds506/507,96each, balanced random two-arm assignment16/motion/seed.
Follow original approach plus full90-row first stationary phase; stop global
simulation at maxphase_stop, keepfinishedmotiontargets atphase_stop. Everymotion
has90phase samples; no fullnativeepisode success claim.

Primary75consecutive inphase: root>=actualframe0+3cm, both historicalforceproxies,
correctfullobject/tableplane clearance>=20mm. Gate>=10%pooled,>=5%eachmotion and
>=5pppooledgain over0. Secondary physicalheight/clearance counts are descriptive
and cannot replace strictforcegate. Save fullposes/forces/actions/targets/progress,
initial qvel/objectvel, actual native objectmass/inertia/flags and gravity. Positive
permits separately frozen teacher/learning feasibility; negative stopsfixedlabel
tracking and returns to grasp/contact construction. No newexternal authorization.
