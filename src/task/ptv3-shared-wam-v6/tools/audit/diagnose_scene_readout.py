"""Bounded real-input scene-head precision/backend differential; no training."""
import argparse
import contextlib
import gc
import json
import math
from pathlib import Path
import sys
import time
import torch
import torch.nn.functional as F

TASK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TASK / 'src'))
from ptv3_shared_wam_v6 import paths
from ptv3_shared_wam_v6.model import SharedWAMV6
from ptv3_shared_wam_v6.data import V6Windows, collate
from ptv3_shared_wam_v5.objectives import forecast_loss as base_loss


def stats(x):
    x = x.detach().float()
    result = dict(shape=list(x.shape), norm=float(x.norm()), rms=float(x.square().mean().sqrt()),
                  absmax=float(x.abs().max()), finite=bool(torch.isfinite(x).all()))
    if x.ndim == 3:
        result['token_centered_rms'] = float((x-x.mean(1, keepdim=True)).square().mean().sqrt())
    return result


def scene_loss(model, scene_z, batch, config):
    mask = batch['scene_valid'] & batch['object_valid'][:, :, None, None]
    target = torch.where(mask[..., None], batch['scene_target'].float(), 0.)
    normalized = model.normalizer.normalize('scene', target)
    error = torch.where(mask[..., None], scene_z.float() - normalized, 0.).square().mean(-1)
    weight = config['motion_floor'] + (1-config['motion_floor']) * torch.sigmoid(
        5 * (target.norm(dim=-1) / config['motion_tau_m'] - 1))
    weight = torch.where(mask, weight, 0.)
    return (error * weight).sum() / weight.sum().clamp_min(1)


def reference_attention(attention, query, context, mask):
    """FP64 reference for this dropout-free, shared-width cross attention."""
    width, heads = attention.embed_dim, attention.num_heads
    q, k, v = [F.linear(x.double(), attention.in_proj_weight[i*width:(i+1)*width].double(),
                        attention.in_proj_bias[i*width:(i+1)*width].double())
               for i, x in enumerate((query, context, context))]
    def split(x):
        return x.view(len(x), -1, heads, width//heads).transpose(1, 2)
    q, k, v = map(split, (q, k, v))
    logits = (q @ k.transpose(-1, -2))/math.sqrt(width//heads)
    logits = logits.masked_fill(mask[:, None, None], -float('inf'))
    result = (logits.softmax(-1) @ v).transpose(1, 2).reshape(len(query), -1, width)
    return F.linear(result, attention.out_proj.weight.double(), attention.out_proj.bias.double()).float()


def fp32_varlen_attention(qkv, cu_seqlens, max_seqlen, dropout_p=0., softmax_scale=None):
    """Process-local FP32 replacement for PTv3's forced-low-precision Flash kernel."""
    if qkv.dtype != torch.float32 or dropout_p != 0:
        raise ValueError('FP32 diagnostic requires float32 QKV and zero dropout')
    lengths = (cu_seqlens[1:]-cu_seqlens[:-1]).long()
    groups = torch.repeat_interleave(torch.arange(len(lengths), device=qkv.device), lengths)
    local = torch.arange(len(qkv), device=qkv.device)-cu_seqlens[:-1].long()[groups]
    padded = qkv.new_zeros((len(lengths), max_seqlen, *qkv.shape[1:]))
    padded[groups, local] = qkv
    query, key, value = padded.unbind(2)
    valid = torch.arange(max_seqlen, device=qkv.device)[None] < lengths[:, None]
    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        output = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2),
            value.transpose(1, 2), attn_mask=valid[:, None, None], dropout_p=0., scale=softmax_scale)
    return output.transpose(1, 2)[groups, local]


def check_varlen_reference():
    """Validate segment packing and gradients against an explicit FP64 loop."""
    lengths = [5, 8, 3]
    cu = torch.tensor([0, 5, 13, 16], device='cuda', dtype=torch.int32)
    values = torch.randn(16, 3, 2, 8, device='cuda', requires_grad=True)
    reference_input = values.detach().double().requires_grad_()
    actual = fp32_varlen_attention(values, cu, max_seqlen=8, softmax_scale=8**-.5)
    result, offset = [], 0
    for length in lengths:
        q, k, v = reference_input[offset:offset+length].unbind(1)
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        result.append((((q @ k.transpose(-1, -2))*8**-.5).softmax(-1) @ v).transpose(0, 1))
        offset += length
    expected = torch.cat(result)
    actual.square().sum().backward()
    expected.square().sum().backward()
    forward_error = float((actual.double()-expected).abs().max())
    gradient_error = float((values.grad.double()-reference_input.grad).abs().max())
    if forward_error > 2e-6 or gradient_error > 5e-6:
        raise ValueError('FP32 variable-length attention reference mismatch')
    return dict(forward_max_error=forward_error, gradient_max_error=gradient_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'manifest', 'stats', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--modes', nargs='+', default=['bf16-default', 'bf16-math', 'fp32',
                        'fp32-attention-only', 'bf16-shared-ln', 'bf16-context-ln', 'fp32-shared-ln'])
    parser.add_argument('--detailed-layers', action='store_true')
    parser.add_argument('--full-fp32', action='store_true',
                        help='Disable external autocast; PTv3 still casts QKV unless --strict-fp32-attention is also set')
    parser.add_argument('--strict-fp32-attention', action='store_true',
                        help='Also replace PTv3 Flash attention with a validated process-local FP32 math reference')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.full_fp32:
        torch.backends.cudnn.allow_tf32 = False
    reference_check = None
    if args.strict_fp32_attention:
        if not args.full_fp32:
            raise ValueError('strict FP32 attention requires --full-fp32')
        sys.path.insert(0, str(paths.ROOT/'third_party/PointWorld'))
        import ptv3.ptv3 as vendor_attention
        import spconv.constants as sparse_constants
        if hasattr(sparse_constants, 'SPCONV_ALLOW_TF32'):
            sparse_constants.SPCONV_ALLOW_TF32 = False
        reference_check = check_varlen_reference()
        vendor_attention.prepare_flash_attn_dtype = lambda values: values.float()
        vendor_attention.flash_attn_varlen_qkvpacked_func = fp32_varlen_attention
    config = json.loads(args.config.read_text())
    normalizer = json.loads(args.stats.read_text())
    dataset = V6Windows(args.manifest, 'train')
    samples, indices = [], []
    for index, _ in dataset.draw(64, 170, 0, 0):
        sample = dataset[index]
        if ((sample['contact_target'] > .25) & sample['interaction_valid']).any():
            samples.append(sample)
            indices.append(index)
        if len(samples) == 4:
            break
    batch = {k: v.cuda() for k, v in collate(samples).items()}
    report = dict(indices=indices, checkpoints=[], python=sys.version, torch=torch.__version__,
                  full_forward_amp=not args.full_fp32, tf32=False,
                  ptv3_attention=('FP32 math reference' if args.strict_fp32_attention else 'vendor forced BF16/FP16 Flash'),
                  varlen_reference_validation=reference_check,
                  scope='scene-head precision/backend differential on detached actual TRAIN activations; no updates')
    begun = time.monotonic()
    for path in args.checkpoints:
        torch.manual_seed(170)
        model = SharedWAMV6(normalizer, config).cuda().train()
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        step = saved['step']
        del saved
        captured = {}
        def capture(module, inputs, kwargs):
            captured.update(query=inputs[0].detach(), context=inputs[1].detach(),
                            mask=kwargs['key_padding_mask'].detach())
        def capture_shared(module, inputs):
            captured['shared'] = inputs[0].detach()
        shared_hook = model.scene_context.register_forward_pre_hook(capture_shared)
        hook = model.scene_readout.register_forward_pre_hook(capture, with_kwargs=True)
        activation_stats = {}
        def inspect_output(name):
            def inspect(module, inputs, output):
                if isinstance(output, tuple):
                    output = output[0]
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = stats(output)
                elif isinstance(output, dict):
                    activation_stats[name] = {k: stats(output[k]) for k in ('tokens', 'hand', 'point_features') if k in output}
            return inspect
        inspected = {'geometry', 'scene_project', 'position', 'hand_project', 'goal', 'event_goal',
                     'hand_encoder', 'geometry.scene_proj', 'geometry.surface_encoder', 'geometry.voxel_fusion',
                     'scene_context'}
        inspected.update('interaction.layers.'+str(i) for i in range(config['shared_layers']))
        if args.detailed_layers:
            inspected.update('interaction.layers.'+str(i)+'.'+name
                             for i in range(config['shared_layers'])
                             for name in ('self_attn', 'norm1', 'norm2', 'linear1', 'linear2'))
        hooks = [module.register_forward_hook(inspect_output(name)) for name, module in model.named_modules()
                 if name in inspected]
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.full_fp32):
            prediction = model(batch)
            _, terms = base_loss(prediction, batch, config, model.normalizer)
        hook.remove()
        shared_hook.remove()
        for extra_hook in hooks:
            extra_hook.remove()
        reference = prediction['scene_normalized'].detach()
        row = dict(checkpoint=str(path), step=step, full_scene_loss=float(terms['scene']),
                   query=stats(captured['query']), context=stats(captured['context']), modes=[],
                   activations=activation_stats,
                   input_stats={k: stats(batch[k]) for k in ('xyz', 'features', 'start_hand',
                       'start_hand_history', 'dense_hand', 'dense_history', 'goal_object_delta')})
        del prediction, terms
        attention = model.scene_readout
        width, heads = attention.embed_dim, attention.num_heads
        # Projected QK statistics use the actual BF16 projection inputs/weights,
        # but FP32 products/softmax for a readable conditioning diagnostic.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=not args.full_fp32):
            q = F.linear(captured['query'], attention.in_proj_weight[:width], attention.in_proj_bias[:width])
            k = F.linear(captured['context'], attention.in_proj_weight[width:2*width], attention.in_proj_bias[width:2*width])
        q = q.view(len(q), -1, heads, width//heads).transpose(1, 2).float()
        k = k.view(len(k), -1, heads, width//heads).transpose(1, 2).float()
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(width//heads)
        logits = logits.masked_fill(captured['mask'][:, None, None], -float('inf'))
        probabilities = logits.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1)
        gap = logits.topk(2, -1).values.diff(dim=-1).abs()
        row['projected_q'] = stats(q)
        row['projected_k'] = stats(k)
        row['attention'] = dict(valid_logit_absmax=float(logits[torch.isfinite(logits)].abs().max()),
                                entropy_mean=float(entropy.mean()),
                                top1_probability_mean=float(probabilities.max(-1).values.mean()),
                                top1_probability_min=float(probabilities.max(-1).values.min()),
                                top_two_gap_median=float(gap.median()))
        del q, k, logits, probabilities, entropy, gap
        for mode in args.modes:
            model.zero_grad(set_to_none=True)
            query = captured['query'].clone().requires_grad_()
            context = captured['context'].clone().requires_grad_()
            normalize_shared = mode in ('bf16-shared-ln', 'fp32-shared-ln')
            full_precision = mode in ('fp32', 'fp32-shared-ln', 'fp32-centered-key', 'fp64-attention-reference')
            torch.cuda.reset_peak_memory_stats()
            kernel = (torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                      enable_mem_efficient=False) if mode == 'bf16-math' else contextlib.nullcontext())
            with kernel, torch.autocast('cuda', dtype=torch.bfloat16, enabled=not full_precision):
                if normalize_shared:
                    context = model.scene_context(F.layer_norm(captured['shared'].float(),
                                                               (captured['shared'].shape[-1],)))
                    context.retain_grad()
                elif mode == 'bf16-context-ln':
                    context = F.layer_norm(context.float(), (context.shape[-1],))
                    context.retain_grad()
                if mode == 'fp64-attention-reference':
                    update = reference_attention(attention, query, context, captured['mask'])
                elif mode in ('bf16-centered-key', 'fp32-centered-key'):
                    valid = (~captured['mask']).float()[..., None]
                    center = (context.float()*valid).sum(1, keepdim=True)/valid.sum(1, keepdim=True)
                    update, _ = attention(query.float(), context.float()-center, context.float(),
                                          key_padding_mask=captured['mask'], need_weights=False)
                elif mode == 'fp32-attention-only':
                    with torch.autocast('cuda', enabled=False):
                        update, _ = attention(query.float(), context.float(), context.float(),
                                              key_padding_mask=captured['mask'], need_weights=False)
                    update = update.to(query.dtype)
                else:
                    update, _ = attention(query if not full_precision else query.float(),
                                          context if not full_precision else context.float(),
                                          context if not full_precision else context.float(),
                                          key_padding_mask=captured['mask'], need_weights=False)
                residual = query + update
                latent = model.scene_norm(residual)
                z = model.scene_future(latent)
                z = z.reshape(len(query), 24, 1, 32, 3).transpose(1, 2)
                loss = scene_loss(model, z, batch, config)
            loss.backward()
            gradients = sorted([(n, float(p.grad.float().norm())) for n, p in model.named_parameters()
                                if p.grad is not None], key=lambda v: v[1], reverse=True)
            norm = math.sqrt(sum(value**2 for _, value in gradients))
            entry = dict(mode=mode, scene_loss=float(loss), scene_head_parameter_grad_norm=norm,
                         query_grad=stats(query.grad), context_grad=stats(context.grad),
                         output_max_difference=float((z-reference).abs().max()),
                         residual_feature_std_min=float(residual.float().std(-1).min()),
                         residual=stats(residual), largest_gradients=gradients[:10],
                         peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            row['modes'].append(entry)
            print(json.dumps(dict(step=step, **entry)), flush=True)
            del query, context, update, residual, latent, z, loss
        report['checkpoints'].append(row)
        report['elapsed_s'] = time.monotonic()-begun
        (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        del model, captured, reference, attention
        gc.collect()
        torch.cuda.empty_cache()
        if report['elapsed_s'] > 300:
            raise TimeoutError('bounded audit exceeded five minutes')


if __name__ == '__main__':
    main()
