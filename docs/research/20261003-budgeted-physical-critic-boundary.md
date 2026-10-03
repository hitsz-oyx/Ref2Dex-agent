# Joint dynamics/Q representations: primary-source boundary

Verified publisher-owned [Zheng et al., DROP: Dynamics-aware representation
learning for offline reinforcement learning, Applied Soft Computing189,
114525(2026)](https://www.sciencedirect.com/science/article/pii/S1568494625018381).
The accessible abstract/introduction describe joint state-action representations,
reward/next-state embedding prediction as auxiliary learning, and BC-related
value regularization. Full implementation/details are not accessible; we do not
reproduce their algorithm, conservatism, results or theorem. Generic dynamics
auxiliary supervision for a Qrepresentation is prior art.

Our new decision concerns the actual collection-budget tradeoff between short
physical episodes and additional complete task episodes for contact-rich
manipulation. Raw observed physical131targets, measured terminal105labels and
jointQauxiliary training are engineering choices. No novelty follows just from
renaming the module Cm or shifting auxiliary loss from V to action-conditionedQ.
A positive actual policy Probe is prerequisite for matchedmulti-training-seed
Validation, a distinctive mechanism and broader journal-quality evidence.
No claim of sample saving unless BOTHphysical collection and task collection
are counted against the explicit equal nativeenv-control-tick budget.
