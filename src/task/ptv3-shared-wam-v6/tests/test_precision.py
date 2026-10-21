from types import MethodType
import copy
import pytest
import torch
from torch import nn
from ptv3_shared_wam_v6.precision import (
    sdpa, varlen_attention, _mha_forward, configure_strict_fp32,
)


def test_ragged_forward_backward_matches_independent_double_reference():
    torch.manual_seed(6)
    qkv = torch.randn(16, 3, 2, 8, requires_grad=True)
    cu = torch.tensor([0, 5, 13, 16], dtype=torch.int32)
    actual = varlen_attention(qkv, cu, 8, backend='math')
    ref = qkv.detach().double().requires_grad_()
    expected = []
    for start, end in zip(cu[:-1], cu[1:]):
        q, k, v = ref[start:end].unbind(1)
        attn = (q.transpose(0, 1) @ k.transpose(0, 1).transpose(-1, -2) / 8**.5).softmax(-1)
        expected.append((attn @ v.transpose(0, 1)).transpose(0, 1))
    expected = torch.cat(expected)
    grad = torch.randn_like(actual)
    actual.backward(grad);expected.backward(grad.double())
    torch.testing.assert_close(actual.double(), expected, atol=8e-7, rtol=2e-5)
    torch.testing.assert_close(qkv.grad.double(), ref.grad, atol=2e-6, rtol=3e-5)
    with pytest.raises(ValueError, match='float32'):
        sdpa(qkv.half(), qkv.half(), qkv.half(), backend='math')


@pytest.mark.parametrize('self_attention', [False, True])
@pytest.mark.parametrize('floating_mask', [False, True])
def test_mha_adapter_preserves_weights_outputs_and_gradients(self_attention, floating_mask):
    torch.manual_seed(7)
    native = nn.MultiheadAttention(16, 2, batch_first=True)
    adapted = copy.deepcopy(native)
    adapted.fp32_backend = 'math';adapted.forward = MethodType(_mha_forward, adapted)
    original = copy.deepcopy(adapted.state_dict())
    q = torch.randn(2, 10 if self_attention else 4, 16, requires_grad=True)
    k = q if self_attention else torch.randn(2, 10, 16, requires_grad=True)
    qr = q.detach().clone().requires_grad_()
    kr = qr if self_attention else k.detach().clone().requires_grad_()
    mask = torch.zeros(2, 10, dtype=torch.bool);mask[:, -3:] = True
    if floating_mask:
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, -float('inf'))
    a, _ = native(q, k, k, key_padding_mask=mask, need_weights=False)
    b, _ = adapted(qr, kr, kr, key_padding_mask=mask, need_weights=False)
    g = torch.randn_like(a);a.backward(g);b.backward(g)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(q.grad, qr.grad, atol=2e-6, rtol=2e-5)
    for (n, p), (_, r) in zip(native.named_parameters(), adapted.named_parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(adapted.state_dict()[n], original[n], rtol=0, atol=0)


def test_legacy_noop_and_ambiguous_precision_rejected():
    module = nn.Linear(2, 2)
    assert configure_strict_fp32(module, {}) == {'strict_fp32': False}
    with pytest.raises(ValueError, match='amp=false'):
        configure_strict_fp32(module, dict(strict_fp32=True, amp=True, tf32=False))
    with pytest.raises(ValueError, match='backend'):
        configure_strict_fp32(module, dict(strict_fp32=True, amp=False, tf32=False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA FP32 efficient backend')
def test_cuda_efficient_fp32_ragged_backward_matches_math():
    torch.manual_seed(8)
    x = torch.randn(16, 3, 2, 8, device='cuda', requires_grad=True)
    y = x.detach().clone().requires_grad_()
    cu = torch.tensor([0, 5, 13, 16], device='cuda', dtype=torch.int32)
    a = varlen_attention(x, cu, 8, backend='efficient')
    b = varlen_attention(y, cu, 8, backend='math')
    g = torch.randn_like(a);a.backward(g);b.backward(g)
    assert a.dtype == torch.float32
    torch.testing.assert_close(a, b, atol=1e-6, rtol=2e-5)
    torch.testing.assert_close(x.grad, y.grad, atol=2e-6, rtol=3e-5)
