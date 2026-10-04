# Bounded extra closure produces a mechanical signal but fails full gates

Run P-20261002-finger-preload-feasibility-r1, code5d7c375, COMPLETED76.99s,
9.43MiB on free GPU5.192 fresh mechanical90-step trajectories, seeds504/505,
48per dose and16per dose/motion. No policy calls; model/RMS unchanged; all189
protected hashes unchanged. Native PD inversion error<=1.193e-7. Correct thin-Y
full-mesh tabletop clearance independently agrees within4.814e-7m; force proxies,
coupling, actual progress and labels independently reconstructed.

| Extra curl rad | Retained75 /48 | Motion0 /16 | Motion1 /16 | Motion2 /16 |
|---|---|---|---|---|
|0|0|0|0|0|
|.05|0|0|0|0|
|.15|6|6|0|0|
|.30|13|8|0|5|

All three positive doses fail the pooled50%and every-motion25%gates. Only.30
passes pooled gain>=25pp versus zero. UNPROMISING; retain ALL arm/motion data,
without larger-dose or selected-seed expansion of this exact static family.
Candidate physical retention on motions0/2 is real recorded evidence, but no
frame0 pickup, learned policy, force-closure or matched training claim follows.
The next distinct question is whether a predetermined approach/lift controller
can reach these mechanical holding states from the original frame0, with contact
geometry and proxy measurement checked separately. Dose.30 is an explicitly
post-hoc candidate for that future prospective screen, not a successful primary.

A figure from full physical root trajectories shows that mean height and strict
75-tick force-proxy conjunction are different quantities; do not replace the
failed primary gate with visually elevated curves. Figure means include16 rows
per motion/dose, rather than cherry-picked successful trajectories.

Paper revision5 now includes completed transfer, one fixed PPO continuation,
static invalid-axis limitation/correction and this prospective closure failure.
12-page PDF review copy,11 JSON-generated tables and2 figures; LaTeX source and
render/source/data hashes retained. Prior revisions remain unchanged. Still NOT
journal ready: novelty, stable self-trained pickup and matched Cm-on/off training
benefit remain absent.
