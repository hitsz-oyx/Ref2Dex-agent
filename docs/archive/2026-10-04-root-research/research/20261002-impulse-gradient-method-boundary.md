# Physical impulse and model-to-actor gradient boundaries

[SVG, Heess et al., NeurIPS2015](https://proceedings.neurips.cc/paper/2015/hash/148510031349642de5ca0c544f31b2ef-Abstract.html)
already differentiates learned dynamics for continuous policy learning and uses
real observed states to limit compounded model errors. Model gradients into an
actor are established, not a new contribution. Gaussian response moments and
action-effect representation boundaries remain in the earlier primary-source
notes. Any future distinct mechanism must be supported by matched policy utility
and comparisons, not new naming or an offline predictor improvement.

New candidate target is normalized NON-GRAVITATIONAL object impulse:
`[m*(v_next-v_current)-m*g*dt]/[m*|g|*dt]`,3world coordinates, dt1/30s.
Native target asset linear damping0.01 is present in base_dexplore_task.py.
This label includes damping and all other nongravitational effects; it is NOT
an exact pairwise hand-contact impulse, contact existence, force closure or
final task value. Conserved momentum algebra is established physics.

CURRENT raw object and five hand-body force vectors normalized by actual weight
give18pre-action channels. Last post-tick force is available at next pre-action
state; never use the force measured after the action being predicted. Nominal
current-force/damping persistence is an approximate prior, not simulator-exact
substep integration. Airborne eligibility is current geometry only.

Hypothesis: action-conditioned impulse prediction captures contact response
better than matched force-aware state-only, force-aware motion/phase action
model, and physical persistence. Positive permits investigating direct actor
gradient coupling; it does not establish utility, novelty or journal readiness.
