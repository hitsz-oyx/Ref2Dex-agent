# Physical response acquisition: primary-source boundary

Read3October2026. This is route selection, not evidence of our method's utility.
The sources below are primary abstracts; no full-paper implementation review
is claimed.

[Plan2Explore, ICML2020](https://proceedings.mlr.press/v119/sekar20a.html)
already uses a learned world model to seek expected future novelty during
exploration and adapts to downstream tasks after exploration. Using a world
model to choose informative training interaction is established.

[CEE-US, NeurIPS2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/98ecdc722006c2959babbdbdeb22eb75-Abstract-Conference.html)
incorporates relational inductive biases into structured world models for
interaction-rich exploration and subsequent model-based object manipulation.
Object structure plus curiosity cannot alone establish novelty here.

[Tactile Intrinsic Motivation, RA-L2021](https://arxiv.org/abs/2102.11051)
uses force-based intrinsic reward and contact-prioritized experience replay
for manipulation. Prioritizing contact-rich transitions is also established;
our aggregate SDK forces are not attributed tactile contact measurements.

Our inference: data acquisition is a distinct use of Cm from this repository's
failed auxiliary/feature/successor actor recipes, but that distinction is not
a new research contribution by itself. A concrete candidate is to prioritize
uncertainty in action CONTRASTS rather than absolute successor predictions.
Common state-dependent prediction variation cancels from a contrast within
each model. This is an algebraic fact, not a proof of calibrated epistemic
uncertainty, informative labels or improved learning. We have not established
whether this exact construction is novel; any eventual novelty claim requires
a more specific primary-source search and strong method comparisons.

Cheapest decision: fixed-label-budget offline acquisition on existing randomized
physical-response data. Bootstrap initial models on whole shared environment
blocks; compare contrast-disagreement, absolute-disagreement and uniform label
acquisition. Train identical final factual models, evaluate identified causal
contrast risk differences on separately acquired, previously viewed IID test
data. Passing permits corrected-physics/native training-data acquisition and
then independent policy learning; it is not yet an online sample-efficiency,
policy, cross-hand or novelty result. All pool/test native interactions were
already paid and must be reported, not relabeled as selected-label savings.
