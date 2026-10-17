# PTv3-Shared-WAM v4

v4 is the interaction-aware successor of WAMv3. It preserves the v3
goal-conditioned shared encoder and hand/scene forecasts, then adds a dynamic
hand--object interaction graph for every forecast step. The graph keeps the
canonical hand point and object query point axes explicit, so a downstream
robot-specific executor can consume contact structure instead of a global
mean-pooled feature.

The current nine-source data contract supplies 32 object query points. v4
therefore uses a documented radial-normal approximation and smooth distance
contact labels. It is an engineering probe, not a dense-surface claim.

`tools/run/train_pretrain.py` accepts `--init-v3 <checkpoint>` for a model-only
warm start. All compatible v3 trunk parameters transfer; the interaction
relation modules are initialized fresh because their tensor contract is new.
The first downstream decision is a matched executor comparison using predicted
`tau` alone versus predicted `tau` plus the structured graph.
