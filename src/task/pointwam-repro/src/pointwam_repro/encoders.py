"""Frozen Mosaic3D point features and Recap-CLIP token states.

PointWAM does not name a Mosaic3D checkpoint. This loader uses the public
full-data PPT checkpoint Mosaic3D trains with Recap-CLIP,
``sc+ar+sc+++ma+st.ckpt``. Its UNet ``out_channels`` is Recap-CLIP's 768-d
embedding size. The text tower width is 1024, and those ``ln_final`` token
states are what Volt projects. The vision tower is loaded with the same
checkpoint and stays unused.

Mosaic3D's own grid is 2 cm and its color map is ``rgb / 127.5 - 1``. The
prompt condition is not named by PointWAM; this loader uses ``ScanNet``, the
first condition of that checkpoint. Both encoders are frozen.
"""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import torch
from torch import nn

MOSAIC3D_REPO = "junhalee/Mosaic3D"
MOSAIC3D_CHECKPOINT = "sc+ar+sc+++ma+st.ckpt"
RECAP_CLIP_REPO = "UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP"
SCENE_FEATURE_DIM = 768
TEXT_TOKEN_DIM = 1024
TEXT_CONTEXT = 128
MOSAIC_GRID_M = 0.02
MOSAIC_CONDITIONS = (
    "ScanNet",
    "ARKitScenes",
    "Matterport3D",
    "Structured3D",
    "ScanNetPP",
)
MOSAIC_CONDITION = "ScanNet"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def mosaic_checkpoint_path() -> Path:
    return repo_root() / "src/task/pointwam-repro/assets/mosaic3d" / MOSAIC3D_CHECKPOINT


def recap_directory() -> Path:
    return repo_root() / "src/task/pointwam-repro/assets/recap-clip"


def normalize_color(rgb: torch.Tensor) -> torch.Tensor:
    """Mosaic3D ``NormalizeColor``: 0–255 RGB becomes about [-1, 1]."""
    if rgb.ndim != 2 or rgb.shape[-1] != 3:
        raise ValueError("rgb must be [N, 3] in 0–255")
    return rgb.to(dtype=torch.float32) / 127.5 - 1.0


def _ensure_mosaic_importable() -> None:
    root = repo_root() / "third_party" / "Mosaic3D"
    if not (root / "src" / "models").is_dir():
        raise FileNotFoundError(f"Mosaic3D source is missing at {root}")
    entry = str(root)
    if entry not in sys.path:
        sys.path.insert(0, entry)


def build_mosaic() -> nn.Module:
    """Build the spunet34c PPT wrapper used by the full-data checkpoint."""
    _ensure_mosaic_importable()
    from src.models.networks.ppt.model import PPT
    from src.models.networks.spunet.spconv_unet_v1m3_pdnorm import SpUNetBase

    backbone = partial(
        SpUNetBase,
        in_channels=3,
        out_channels=SCENE_FEATURE_DIM,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        conditions=MOSAIC_CONDITIONS,
        zero_init=True,
        norm_decouple=True,
        norm_adaptive=True,
        norm_affine=True,
    )
    return PPT(backbone, conditions=list(MOSAIC_CONDITIONS), context_channels=256)


def _net_weights(blob: object) -> dict[str, torch.Tensor]:
    if not isinstance(blob, dict) or "state_dict" not in blob:
        raise ValueError("Mosaic3D checkpoint has no Lightning state_dict")
    weights = {
        key[len("net."):]: value
        for key, value in blob["state_dict"].items()
        if key.startswith("net.")
    }
    if not weights:
        raise ValueError("Mosaic3D checkpoint has no net weights")
    return weights


def load_mosaic(checkpoint: Path | None = None, device: torch.device | str = "cpu") -> nn.Module:
    """Load frozen Mosaic3D point features. Missing checkpoint is an error."""
    path = Path(checkpoint) if checkpoint is not None else mosaic_checkpoint_path()
    if not path.is_file() or path.stat().st_size < 2_000_000_000:
        raise FileNotFoundError(f"complete Mosaic3D checkpoint is required at {path}")
    model = build_mosaic()
    blob = torch.load(path, map_location="cpu")
    weights = _net_weights(blob)
    own = model.state_dict()
    selected = {key: value for key, value in weights.items() if key in own}
    missing = [key for key in own if key not in selected]
    if missing:
        raise RuntimeError(f"Mosaic3D checkpoint is missing {missing[:8]}")
    model.load_state_dict(selected, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def mosaic_point_features(
    model: nn.Module,
    xyz: torch.Tensor,
    rgb: torch.Tensor,
    condition: str = MOSAIC_CONDITION,
) -> torch.Tensor:
    """Return one 768-d feature per scene point. ``xyz`` is meters."""
    if condition not in MOSAIC_CONDITIONS:
        raise ValueError(f"unknown Mosaic3D condition {condition}")
    if xyz.ndim != 2 or xyz.shape[-1] != 3 or xyz.shape[0] != rgb.shape[0]:
        raise ValueError("xyz and rgb must describe the same points")
    device = next(model.parameters()).device
    batch = {
        "coord": xyz.to(device=device, dtype=torch.float32),
        "feat": normalize_color(rgb).to(device),
        "batch": torch.zeros(xyz.shape[0], dtype=torch.long, device=device),
        "grid_size": MOSAIC_GRID_M,
        "condition": [condition],
    }
    was_training = model.training
    model.eval()
    with torch.no_grad():
        point = model(batch)
        features = point.sparse_conv_feat.features[point.v2p_map]
    if was_training:
        model.train()
    if features.shape != (xyz.shape[0], SCENE_FEATURE_DIM):
        raise RuntimeError(f"unexpected Mosaic3D features {tuple(features.shape)}")
    return features


def load_recap_text(directory: Path | None = None, device: torch.device | str = "cpu"):
    """Load the frozen Recap-CLIP text tower and its local tokenizer."""
    root = Path(directory) if directory is not None else recap_directory()
    weight = root / "open_clip_pytorch_model.bin"
    if not weight.is_file() or weight.stat().st_size < 2_000_000_000:
        raise FileNotFoundError(f"complete Recap-CLIP weights are required at {weight}")
    from open_clip.factory import load_checkpoint
    from open_clip.model import CLIP
    from open_clip.tokenizer import HFTokenizer

    config = json.loads((root / "open_clip_config.json").read_text())["model_cfg"]
    model = CLIP(**config)
    load_checkpoint(model, str(weight))
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokenizer = HFTokenizer(str(root), context_length=TEXT_CONTEXT, strip_sep_token=True)
    return model, tokenizer


def recap_token_features(model: nn.Module, tokenizer, text: str, device: torch.device | str) -> torch.Tensor:
    """Return non-padding ``ln_final`` token states, shape ``[L, 1024]``."""
    if not str(text).strip():
        raise ValueError("instruction text is empty")
    token_ids = tokenizer(str(text)).to(device)
    cast = model.transformer.get_cast_dtype()
    hidden = model.token_embedding(token_ids).to(cast)
    hidden = hidden + model.positional_embedding.to(cast)
    hidden = model.transformer(hidden, attn_mask=model.attn_mask)
    hidden = model.ln_final(hidden)
    keep = token_ids[0] != 0
    tokens = hidden[0, keep]
    if tokens.ndim != 2 or tokens.shape[-1] != TEXT_TOKEN_DIM or tokens.shape[0] < 1:
        raise RuntimeError(f"unexpected text tokens {tuple(tokens.shape)}")
    return tokens
