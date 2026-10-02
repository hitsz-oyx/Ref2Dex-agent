# P-20261003-cm-mve-object-projection

Family: Cm decision interface
Type: Decision Probe
Status: COMPLETED — `UNCLEAR` support boundary, with a negative native utility screen

This Probe kept Cm as a physical consequence model and changed only the MVE
continuation representation.  For each candidate, Cm predicted object
translation/velocity/contact/events; the current hand joint state and object
orientation were retained before applying the frozen value continuation.  No
future state, model update, actor update, or PPO was available to the selector.

The offline held-transition audit used 209,788 rows from the existing physical
model collection.  Object-projected MVE improved row-level target error/ranking
over direct-Q, but the gain was small and did not improve episode-level ranking.
The fresh native panel had 127 complete windows and changed 25 object-MVE
actions.  Its lower90 native height and local-reward differences versus Cup
were negative, and its random-arm action ranking was weaker than direct-Q.

The predeclared action-interface gates therefore do not pass.  The result does
not refute Cm's physical prediction; it closes this MVE-to-action-value
conversion and leaves the North-star policy utility open.  Full evidence is in
[`P-20261003-cm-mve-object-projection-results.json`](P-20261003-cm-mve-object-projection-results.json).
