"""Supervised global-mean loss for unequal local DDP batch sizes."""
def global_mean_loss(prediction, target, scale, global_batch, world_size):
    local = len(prediction)
    if not 0 < local <= global_batch or world_size < 1:
        raise ValueError('positive local/global BC batch required')
    # DDP averages gradients over ranks, not examples.
    return ((prediction-target)/scale).abs().mean() * (world_size*local/global_batch)
