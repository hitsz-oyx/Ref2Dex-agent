# P-20261003-cm-candidate-coverage

Family: Cm decision interface
Type: Data sufficiency Decision Probe
Status: COMPLETED — `UNPROMISING` for ordinary candidate expansion

This Probe tested whether the negative Cm-relative action-value screen was mainly
caused by too few motion/start groups. The frozen direct-Q, physical Cm ensemble,
candidate contract, split rule, and labels were held fixed. Only candidate panels
with a new coverage salt were collected; no model or policy was trained.

The first repeat added 148 rows but no new motion/start groups: the combined panel
grew from 287 to 435 rows while remaining at 25 groups. Its held adapter lower90
was `+2.417 mm` for state/action features, but `-21.861 mm` after adding Cm physical
features. This is row expansion without decision-support expansion.

Two further coverage salts added seven complete groups in r3 and thirteen in r4.
The final complete panel has 411 rows across 45 motion/start groups, with arm
support `[55, 58, 55, 50, 57, 54, 37, 45]`; the predeclared support gate passes.
On this panel, the frozen adapter gives state/action lower90 `-29.187 mm` and
state/action+Cm lower90 `-22.162 mm`. The mean Cm increment is only `+1.189 mm`
and its held lower90 remains negative. The raw uncertainty fallback and raw top
selector also have negative lower90 score deltas (`-2.02` and `-11.54 mm`).

The coverage hypothesis is therefore not supported: increasing rows and adding
motion/start support did not produce a positive held Cm value conversion. Close
ordinary candidate-data expansion under this contract. Do not start PPO or scan
adapter thresholds; the next useful route must change the value representation or
decision target.

The two incomplete r3 launches and the first r4 path typo are retained as execution
manifests, but are excluded from the complete scientific panel. Full aggregate
values are in [`P-20261003-cm-candidate-coverage-results.json`](P-20261003-cm-candidate-coverage-results.json).
