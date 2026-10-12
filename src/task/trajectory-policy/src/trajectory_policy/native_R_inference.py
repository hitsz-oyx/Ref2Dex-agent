"""Keep qualified PD-R inference on its calibrated batched native-runtime path."""

def calibrated_PD_forward(model, state, tau):
    # On pinned Torch2.0.1/cu118, the retained PD MLP's singleton path differs
    # materially from both calibrated batch2 and Torch2.4.1. Duplicate only the
    # current query to use the qualified path; no state/trajectory substitution.
    if len(state) == 1:
        return model(state.repeat(2,1), tau.repeat(2,1,1))[:1]
    return model(state,tau)
