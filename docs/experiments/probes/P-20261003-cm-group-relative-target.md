# P-20261003-cm-group-relative-target

Family: Cm decision interface
Type: Action-relative target Decision Probe
Status: COMPLETED — `UNPROMISING`, close this offline action-target route

This Probe tested whether changing the target from an absolute candidate score to a
motion/start-group-centered realized H10 score would expose a usable Cm action
advantage. The candidate panel, frozen Cm checkpoint, uniform `p=1/8` propensity,
and no-future-state input contract were held fixed. The final panel has 411 rows,
45 motion/start groups, and arm support `[55, 58, 55, 50, 57, 54, 37, 45]`.

A single fit/held split looked positive for the group-relative adapter: the held
IPW score lower90 was `+0.675 mm` and held demeaned Spearman was `0.623`. That
signal did not survive the predeclared five-fold group cross-fit. The group-relative
adapter reached held demeaned Spearman `0.704`, but its cross-fit IPW score delta
was mean `-11.416 mm`, lower90 `-39.179 mm`; retention/contact/clearance lower90
were `-0.315/-0.344/-0.320`. The absolute Cm adapter was also negative
(mean `-11.772 mm`, lower90 `-37.258 mm`). The fixed split was therefore treated
as a split artifact, not action utility.

Close this target change. Do not collect ordinary candidate data, tune the target,
or start PPO from this result. Any future action interface needs a new source of
identification or a different experimental control. Full cross-fit values and
fold support are in [`P-20261003-cm-group-relative-target-results.json`](P-20261003-cm-group-relative-target-results.json).
