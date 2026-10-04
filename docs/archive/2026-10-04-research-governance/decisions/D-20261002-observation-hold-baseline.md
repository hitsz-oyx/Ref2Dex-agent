# Decision: test a trained observation policy before more model-guided learning

Question: can a policy trained from our verified controller independently reach
and maintain physical lift from frame0, rather than relying on teacher actions?
Support witness49cf8b3 is PROMISING for motion1, with unchanged32/32geometric
holding105 throughout the proposed stronger window (post-hoc feasibility check).
Motion0/2 remain unproved; all three are retained in fit/evaluation reports.

Choose one scratch BC policy as a cheap P0 initializer check, not final Cm utility.
One teacher seed510,96env,202steps, collect all current70-feature/action rows.
Features: nativeq18,dq18,objectroot13,currentforceproxies2,plannedreferenceq18 at
min(progress+1,plateau_stop),clippedreferenceprogress/plateau_stop1. No future
physical states, motion identity, teacher action or privileged outcome in input.
The fixed reference plan is explicitly available to both teacher and policy.

Random initialization seed734, existing DExploreBcPolicy512/256/128ReLU/Tanh,
2000 fixed Adam updates,lr.001,batch512,gradnormclip10. Normalize70features by
FIT data only,stdfloor.001; clipnormalizedcontext to[-10,10]. Loss divides action
errors by fixed scales [.005]*3+[.05]*3+[.1]*12 before mean-square. Six hardware
nullcommands canonical0 for both teacher andpolicy. Finalonly checkpoint; no
optimizer/official/source actor weights imported into our trained policy.

Fresh evaluation511/512,96each,202ticks, no teacher fallback or actionselection.
NEW prospective physical primary: all105ticks from plateau_stop-74 through
plateau_stop+30 have root>=actualframe0+3cm and fullmesh tableclearance>=20mm.
This entails75phaseholding ticks plus30postphase drop checking with explicit
holding reference plan. Gravity/mass/threeactors and no objectstate writes checked.
Force75 remains secondary; all historical failed gates and rewards unchanged.

Initial task feasibility gate: knownmotion1 physical105>=50%pooled and>=25%each
newseed. Reportallmotions; PROMISING means only this policy initializer supports
next matched learning design, not the journal objective or general manipulation.
Failure ends this exact BC fit without more epochs/seed/modelsize selection;
consider distinct on-policy aggregation if action regression fails via covariateshift.
One admittedfreeGPU,<=1800s/1GiB, no new external authorization needed.
