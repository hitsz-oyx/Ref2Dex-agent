# Transformer residual amplification reference

Primary sources: [paper](https://arxiv.org/abs/2004.08249),
[author code and explanation](https://github.com/LiyuanLucasLiu/Transformer-Clinic).
Read the abstract and repository explanation, not the full paper.

The authors identify residual-branch amplification of small parameter changes
as a Transformer stability mechanism and propose adaptive initialization.
This motivates inspecting activations and parameter perturbations, rather than
assuming that gradient imbalance alone explains failure.

Our WAM FFN rollback/activation measurements are local evidence, not a
reproduction of this paper. They do not prove that Admin is the right repair,
that pre-norm always fails, or that a missing final normalization alone caused
our particular trajectory. No external code was installed or executed.
