"""Explicit function-preserving warm start for a wider/deeper shared trunk."""
import torch


@torch.no_grad()
def expand_forecaster(target, state, source_config):
    source_width = int(source_config.get('interaction_width',128))
    source_layers = int(source_config.get('interaction_layers',4))
    width = target.interaction_width
    if width < source_width or width % source_width or len(target.interaction.layers)<source_layers:
        raise ValueError('expansion requires divisible width and non-decreasing depth')
    repeat = width//source_width
    result = target.state_dict()
    copied = []
    for key,value in state.items():
        if key not in result:
            raise ValueError('unexpected source parameter: '+key)
        if not key.startswith('interaction.'):
            if result[key].shape != value.shape:
                raise ValueError('only the shared interaction trunk may expand: '+key)
            result[key] = value
        elif key.endswith('self_attn.in_proj_weight'):
            result[key] = value.reshape(3,source_width,source_width).repeat(1,repeat,repeat).reshape(3*width,width)/repeat
        elif key.endswith('self_attn.in_proj_bias'):
            result[key] = value.reshape(3,source_width).repeat(1,repeat).reshape(3*width)
        elif value.ndim == 2:
            result[key] = value.repeat(repeat,repeat)/repeat
        else:
            result[key] = value.repeat(repeat)
        if result[key].shape != target.state_dict()[key].shape:
            raise ValueError('invalid expanded shape: '+key)
        copied.append(key)
    expected = {key for key in result if not key.startswith('interaction.layers.')
                or int(key.split('.')[2])<source_layers}
    if set(copied) != expected:
        raise ValueError('missing source weights during expansion')
    target.load_state_dict(result,strict=True)
    for layer in list(target.interaction.layers)[source_layers:]:
        # Independent new internal features; zero residual outputs start as
        # identity and receive nonzero output-projection gradients immediately.
        layer.self_attn._reset_parameters()
        layer.linear1.reset_parameters()
        layer.self_attn.out_proj.weight.zero_()
        layer.self_attn.out_proj.bias.zero_()
        layer.linear2.weight.zero_()
        layer.linear2.bias.zero_()
    return dict(method='integer-width-replication-and-identity-depth.v1',
        source_width=source_width,target_width=width,source_layers=source_layers,
        target_layers=len(target.interaction.layers),copied_tensors=len(copied),optimizer_reset=True)
