"""Opt-in FP32 attention without changing vendor code or checkpoint keys.

The PointWorld backend otherwise casts QKV to BF16 even outside autocast.
Efficient SDPA supports float32 on our GPUs; no half-precision fallback is allowed.
"""
from types import MethodType
import torch
from torch import nn
from torch.nn import functional as F


def sdpa(q, k, v, mask=None, scale=None, backend='efficient'):
    if any(x.dtype != torch.float32 for x in (q, k, v)):
        raise ValueError('strict FP32 attention requires float32 QKV')
    if backend not in ('efficient', 'math'):
        raise ValueError('explicit efficient or math FP32 backend required')
    if mask is not None:
        # Efficient CUDA requires aligned bias strides, including short sequences.
        width = mask.shape[-1]
        storage = q.new_zeros((*mask.shape[:-1], (width + 7) // 8 * 8))
        bias = storage[..., :width]
        if mask.dtype == torch.bool:
            bias.masked_fill_(~mask, -float('inf'))  # SDPA True means allowed.
        else:
            bias.copy_(mask)
        mask = bias
    with torch.backends.cuda.sdp_kernel(enable_flash=False,
            enable_math=backend == 'math', enable_mem_efficient=backend == 'efficient'):
        if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 1):
            # Torch2.0 lacks the scale keyword. Preserve SDPA's default scale
            # and apply a custom ratio only when one was actually requested.
            default_scale = q.shape[-1] ** -.5
            if scale is not None and scale != default_scale:
                q = q * (scale / default_scale)
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
                                             dropout_p=0., scale=scale)


def varlen_attention(qkv, cu_seqlens, max_seqlen, scale=None, backend='efficient'):
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    groups = torch.repeat_interleave(torch.arange(len(lengths), device=qkv.device), lengths)
    local = torch.arange(len(qkv), device=qkv.device) - cu_seqlens[:-1].long()[groups]
    padded = qkv.new_zeros(len(lengths), max_seqlen, *qkv.shape[1:])
    padded[groups, local] = qkv
    q, k, v = padded.unbind(2)
    valid = torch.arange(max_seqlen, device=qkv.device)[None] < lengths[:, None]
    result = sdpa(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                  valid[:, None, None], scale, backend)
    return result.transpose(1, 2)[groups, local]


def _serialized_forward(self, point):
    pad, unpad, cu = self.get_padding_and_inverse(point)
    order = point.serialized_order[self.order_index][pad]
    inverse = unpad[point.serialized_inverse[self.order_index]]
    qkv = self.qkv(point.feat)[order].reshape(-1, 3, self.num_heads, self.channels // self.num_heads)
    feat = varlen_attention(qkv, cu, self.patch_size, self.scale, self.fp32_backend)
    point.feat = self.proj_drop(self.proj(feat.reshape(-1, self.channels)[inverse]))
    return point


def _mha_forward(self, query, key, value, key_padding_mask=None, need_weights=True,
                 attn_mask=None, average_attn_weights=True, is_causal=False):
    if need_weights or attn_mask is not None or is_causal:
        raise ValueError('strict WAM FP32 MHA supports only its noncausal key-padding contract')
    width = self.embed_dim
    weights = self.in_proj_weight.chunk(3)
    biases = self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None,) * 3
    projected = [F.linear(x, w, b) for x, w, b in zip((query, key, value), weights, biases)]
    q, k, v = [x.reshape(x.shape[0], x.shape[1], self.num_heads, width // self.num_heads).transpose(1, 2)
               for x in projected]
    mask = key_padding_mask
    if mask is not None:
        mask = ((~mask) if mask.dtype == torch.bool else mask)[:, None, None]
    result = sdpa(q, k, v, mask, backend=self.fp32_backend)
    result = result.transpose(1, 2).reshape(query.shape[0], query.shape[1], width)
    return self.out_proj(result), None


def configure_strict_fp32(model, config):
    if not config.get('strict_fp32', False):
        return dict(strict_fp32=False)
    if config.get('amp', True) or config.get('tf32', True):
        raise ValueError('strict_fp32 requires amp=false and tf32=false')
    backend = config.get('fp32_attention_backend')
    if backend not in ('efficient', 'math'):
        raise ValueError('explicit FP32 attention backend required')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Inference must not bypass the patched attention via Transformer fastpath.
    if hasattr(torch.backends, 'mha'):
        torch.backends.mha.set_fastpath_enabled(False)
    else:
        # Torch2.0 predates the global fastpath switch. This flag only selects
        # its fused encoder path; the actual GELU callable remains unchanged.
        for layer in model.modules():
            if isinstance(layer, nn.TransformerEncoderLayer):
                layer.activation_relu_or_gelu = 0
    import spconv.constants as sparse_constants
    sparse_constants.SPCONV_ALLOW_TF32 = False
    counts = dict(serialized_attention=0, dense_attention=0)
    for module in model.modules():
        if module.__class__.__name__ == 'SerializedAttention':
            if module.attn_drop != 0:
                raise ValueError('strict FP32 adapter requires zero attention dropout')
            module.fp32_backend = backend
            module.forward = MethodType(_serialized_forward, module)
            counts['serialized_attention'] += 1
        elif isinstance(module, nn.MultiheadAttention):
            if (not module.batch_first or not module._qkv_same_embed_dim or module.dropout != 0
                    or module.add_zero_attn or module.bias_k is not None or module.bias_v is not None):
                raise ValueError('unsupported FP32 MHA contract')
            module.fp32_backend = backend
            module.forward = MethodType(_mha_forward, module)
            counts['dense_attention'] += 1
    return dict(strict_fp32=True, attention_backend=backend, attention_dtype='torch.float32',
                matmul_tf32=False, cudnn_tf32=False, spconv_tf32=False, **counts)


class ActivationMonitor:
    """Only observe existing forwards on logged training steps; retain no graphs."""
    def __init__(self, model):
        self.enabled = False
        self.values = {}
        def capture(name):
            def hook(module, inputs, output):
                if self.enabled:
                    x = output.detach()
                    self.values[name] = float(x.float().square().mean().sqrt())
            return hook
        self.handles = [model.interaction.layers[-1].register_forward_hook(capture('shared_rms')),
                        model.scene_context.register_forward_hook(capture('scene_context_rms'))]
