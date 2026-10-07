"""Process-local, integer-exact Hilbert fusion for the frozen PTv3 adapter.

Preserves all four spatial orders, time identity, model weights and RNG draws.
The bit operations below implement the pinned vendor's boolean-array encoder;
they do not substitute a differently oriented Hilbert curve.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _spread(value):
    value = (value | (value << 32)) & 0x1f00000000ffff
    value = (value | (value << 16)) & 0x1f0000ff0000ff
    value = (value | (value << 8)) & 0x100f00f00f00f00f
    value = (value | (value << 4)) & 0x10c30c30c30c30c3
    return (value | (value << 2)) & 0x1249249249249249


@triton.jit
def _hilbert_kernel(coords, codes, count, depth: tl.constexpr,
                    block: tl.constexpr):
    i = tl.program_id(0) * block + tl.arange(0, block)
    mask = i < count
    x = tl.load(coords + i * 3, mask, 0).to(tl.int64)
    y = tl.load(coords + i * 3 + 1, mask, 0).to(tl.int64)
    z = tl.load(coords + i * 3 + 2, mask, 0).to(tl.int64)
    for bit in tl.static_range(depth):
        q = 1 << (depth - 1 - bit)
        lower = q - 1
        x = tl.where((x & q) != 0, x ^ lower, x)
        on = (y & q) != 0
        x = tl.where(on, x ^ lower, x)
        exchange = tl.where(on, 0, (x ^ y) & lower)
        x = x ^ exchange
        y = y ^ exchange
        on = (z & q) != 0
        x = tl.where(on, x ^ lower, x)
        exchange = tl.where(on, 0, (x ^ z) & lower)
        x = x ^ exchange
        z = z ^ exchange
    # The original flattens gray bits in (bit, x/y/z) order, then computes
    # prefix XOR. Integer spreading and Gray decoding are the same operation.
    gray = (_spread(x) << 2) | (_spread(y) << 1) | _spread(z)
    gray = gray ^ (gray >> 1)
    gray = gray ^ (gray >> 2)
    gray = gray ^ (gray >> 4)
    gray = gray ^ (gray >> 8)
    gray = gray ^ (gray >> 16)
    gray = gray ^ (gray >> 32)
    tl.store(codes + i, gray, mask)


@torch.inference_mode()
def fused_hilbert_encode(coords, num_dims=3, num_bits=16):
    """Pinned PTv3 encode signature; CPU retains the upstream implementation."""
    if num_dims != 3 or not 1 <= num_bits <= 16 or coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError('expected Nx3 grid coordinates with depth1..16')
    if coords.dtype not in (torch.int32, torch.int64):
        raise ValueError('integer grid coordinates required')
    if not coords.is_cuda:
        from ptv3.serialization.hilbert import encode
        return encode(coords, num_dims=3, num_bits=num_bits)
    if not len(coords):
        raise ValueError('nonempty grid required')
    coords = coords.contiguous()
    codes = torch.empty(len(coords), dtype=torch.int64, device=coords.device)
    with torch.cuda.device(coords.device):
        _hilbert_kernel[(triton.cdiv(len(coords), 256),)](coords, codes, len(coords), num_bits, 256)
    return codes.squeeze()


def install_fused_hilbert():
    """Patch this process only; caller must record this module in run identity."""
    from ptv3.serialization import default
    original = default.hilbert_encode_
    default.hilbert_encode_ = fused_hilbert_encode
    return original


def restore_hilbert(original):
    from ptv3.serialization import default
    default.hilbert_encode_ = original
