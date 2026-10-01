# P-20261002-frame0-tracking-feasibility

Decision/Blocker, no policy calls or neural training. Fixed referencePD tracking
fromnativeframe0, fresh506/507two96-envpanels,32/motion. Within eachmotionpermute
16copies ofarm0/1withprivate CPU seed eval+8000beforephysics. Arm0curladdition0,
arm1rampedaddition.30rad chosenposthocfromfailedpreviousstaticdoseProbe.

Target p+1labelq clipped in time at firstplateau_stop; fingerparents6/8/10/12/15
clamped againstnativeown/dependentbounds then coupled, thumb yaw14 clamped to its
ownboundedrange. Otherwristcoordinatesunchanged. Closurefractionclamp((p+1-(first
liftstart-15))/15,0,1). Independentfullphysicsinitialstatesbeforeassignment.
Noobjectstatewritesafterinitialreset, forceinjection, actionclipping, actor/RMS
updates, earlytermination or adaptivekappa. NativePDtargeterror<=1e-5 orstop.

Globalmaxfirstplateau_stop162ticks, actualprogress+1verified; each motion90physical
phase samples, includingallarmrows. Nointentionalrelease/fullnativeepisodeclaim.
Held75criterion:root>=ownactualframe0+3cm,bothnativeforceproxies>.1, ALLobjectmesh
vertices>=20mmabovecorrectthin-Ytabletopplane. ProspectivePROMISINGifarm1pooled
>=10%,eachmotion>=5%,and>=5pppooledgainoverarm0. OtherwiseUNPROMISING. Allarm/motion
results retained. Secondaryheight/clearance-only75 neverreplacesprimarygate.

Save fullobjectroot,nativeq,5handforcevectors,objectforce,action,target,progress,
initialvelocities,motion,assignment/ramplabels,tablepose,nativemass/inertia/flags,
simgravity. IndependentlyNumPyauditworld-upfullmeshclearance,force,phaseandlongest
labels. SourceactorloadedONLYplayerconstruction,get_actionforbidden.<=600s/128MiB,
oneadmittedGPU;stopinputdrift/nonfinite/shape/end/clip/inversePD/budgetfailure.
Thisisamechanicalpickupfeasibilityscreen,notlearnedpolicy,forceclosureorValidation.
