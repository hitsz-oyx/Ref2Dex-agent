# PointWorld FFN versus the WAM shared Transformer

Sources: clean local PointWorld0548482, verified against official
[PTv3](https://github.com/NVlabs/PointWorld/blob/main/ptv3/ptv3.py) and
[model setup](https://github.com/NVlabs/PointWorld/blob/main/pointworld/base.py).
No third-party source was edited. This is a code comparison, not paper evidence
that one training recipe is superior.

## What is actually the same

PointWorld PTv3's FFN is a two-linear GELU MLP, width expansion4, pre-norm residual,
with MLP dropout0 in release. Our extra shared Transformer uses the same core
mapping. A local CPU FP64 operator smoke copied Linear/LayerNorm weights between
the actual PointWorld MLP and Torch TransformerEncoderLayer: forward, input
gradient and all FFN/LayerNorm parameter gradients match exactly (max error0).
CPU is used for an algebraic contract check, not model training or inference.
Evidence: outputs/ptv3-shared-wam-v6/pointworld-ffn-audit-20261021-r1/report.json.

## Differences around the FFN

| Item | PointWorld release PTv3 | WAM extra shared stack |
| --- | --- | --- |
| Residual stochastic depth | Scheduled DropPath up to0.3 | None; ordinary dropout0 |
| LayerScale | Supported but defaultNone, not enabled by builder | None |
| Blocks | Hierarchical serialized point blocks, CPE and normalized pooling/unpooling | Additional12 global shared blocks at width768 |
| Linear initialization | Independently initialized blocks; truncated normal std0.02, bias0 | Torch defaults; encoder deep-copies one initialized layer |
| Final FFN residual | Pre-norm; no forced post-residual normalization | Pre-norm; same limitation |

Our PTv3 geometry backbone already imports this vendor implementation; its
configured DropPath cap is0.1. The diagnosed scale growth was in the extra
shared interaction.layers, not that PTv3 backbone. PointWorld's dynamics
predictor uses PTv3 plus a small flow MLP, not our added shared768/12 stack.
Hierarchical normalization and stochastic depth are concrete differences,
not evidence that either alone prevents the observed failure.

## Initialization finding and its limits

Fresh Torch2.4.1 TransformerEncoder layers have equal initial FFN values but
separate parameter storage. Verified locally; official
[documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.TransformerEncoder.html)
also recommends explicit initialization of the cloned layers. WAM construction
does not perform per-layer reinitialization after the clone.

The actual pretrained source4371 layers are already different: layer0 versus
1/3/5/11 linear2 flattened cosines0.729/0.516/0.460/0.415. Historical initial
alignment is a candidate concern, not a demonstrated cause of the new
fine-tuning instability. Never reinitialize the current learned checkpoint as
a purported compatible fix. Evidence: same audit directory/initialization.json.

## Consequence for the diagnosis

The earlier emphasis on missing final normalization was too strong as a
standalone architectural explanation: PointWorld also uses pre-norm residual
FFNs. A final normalization remains a candidate intervention, not an established
missing ingredient or proven root cause. The measurements establish that the
shared FFN/residual stream amplified during one trajectory; they do not establish
that the basic GELU MLP formulation is defective.

The smallest useful next structural comparison retains joint21/goal/events,
hand/scene/contact heads and geometry objective, changing only shared residual
stochastic depth in a separately recorded run. Independent initialization is
relevant to new models; shared-rate/freeze controls are relevant to existing
checkpoints. LayerScale would be a separate proposed extension, not something
we can claim PointWorld release already uses. No model/config/training changes
were made during this inspection.
