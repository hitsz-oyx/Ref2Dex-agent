"""Explicit local assets; no implicit model download or CUDA at import time."""
import importlib
import inspect
import json
from pathlib import Path
import sys
import tempfile
import numpy as np


class AnnotatedMasks:
    """Reviewable current/label masks indexed by native frame, per episode key.

    annotation JSON: {episodes: {key: {target_name, current_verified,
      frames: {"3": {target: "masks/t3.png", hand: "masks/h3.png"}}}}}
    Future masks are read ONLY by label_masks, never by select_current.
    """
    def __init__(self, path):
        self.root = Path(path).resolve().parent
        self.meta = json.loads(Path(path).read_text())

    def _read(self, name, size):
        import cv2
        from .io import confined_path
        mask = cv2.imread(str(confined_path(self.root, name)), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError("mask cannot be read: " + name)
        return cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST) > 0

    def select_current(self, key, tick, rgb, native):
        record = self.meta["episodes"][key]
        entry = record["frames"][str(tick)]
        if not record.get("current_verified") or not record.get("target_name"):
            raise ValueError("current target part needs explicit visual confirmation")
        return dict(target=self._read(entry["target"], rgb.shape[:2]),
                    hand=self._read(entry["hand"], rgb.shape[:2]),
                    name=record["target_name"], current_verified=True,
                    method="explicit-frame-mask")

    def label_masks(self, key, frame_ids, shape):
        record = self.meta["episodes"][key]
        masks = []
        for frame in frame_ids:
            entry = record["frames"].get(str(int(frame)))
            if entry is None:
                return None  # Missing identity evidence cannot become training data.
            masks.append(self._read(entry["target"], shape) & ~self._read(entry["hand"], shape))
        return np.stack(masks)


class CoTrackerBackend:
    def __init__(self, vendor, checkpoint, device="cuda:0"):
        import torch
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        sys.path.insert(0, str(Path(vendor).resolve()))
        from cotracker.predictor import CoTrackerPredictor
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("production tracking uses GPU")
        self.model = CoTrackerPredictor(checkpoint=str(checkpoint), offline=True, window_len=60).eval().to(self.device)

    def track(self, rgb, uv):
        import torch
        if len(rgb) != 25 or not 1 <= len(uv) <= 128:
            raise ValueError("25 native frames and bounded sparse queries required")
        video = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float().to(self.device)
        queries = torch.from_numpy(np.column_stack([np.zeros(len(uv), np.float32), uv]))[None].to(self.device)
        with torch.inference_mode():
            tracks, visible = self.model(video, queries=queries, backward_tracking=False)
        return tracks[0].float().cpu().numpy(), visible[0].cpu().numpy().astype(bool)


class OmegaDepthBackend:
    """Short-context Omega depth; native metric scale is applied by the pipeline.

    Landscape RGB only: the vendor preprocessing must be a pure resize, without
    crops/padding. This fails closed for unsupported layouts instead of guessing K.
    """
    def __init__(self, vendor, checkpoint, scratch, device="cuda:0", resolution=512):
        import torch
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        self.scratch = Path(scratch).resolve()
        if self.scratch == Path("/tmp") or Path("/tmp") in self.scratch.parents:
            raise ValueError("scratch must be in project tmp, not /tmp")
        self.scratch.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(Path(vendor).resolve()))
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        self.preprocess = load_and_preprocess_images
        self.resolution = resolution
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("production depth inference uses GPU")
        self.model = VGGTOmega().eval()
        self.model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        self.model.to(self.device)

    def predict(self, rgb, frame_ids, role):
        import cv2
        import torch
        if len(rgb) != len(frame_ids) or len(rgb) > 25 or role not in ("observation", "label"):
            raise ValueError("bounded explicit depth context required")
        height, width = rgb.shape[1:3]
        if height > width:
            raise ValueError("portrait input requires explicit crop/pad geometry adapter")
        with tempfile.TemporaryDirectory(prefix="omega-", dir=self.scratch) as folder:
            paths = []
            for i, frame in enumerate(rgb):
                path = Path(folder) / (str(i) + ".png")
                if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise IOError("failed to write owned depth scratch")
                paths.append(str(path))
            options = {}
            signature = inspect.signature(self.preprocess).parameters
            if "image_resolution" in signature:
                options["image_resolution"] = self.resolution
            if "mode" in signature:
                # Omega exposes balanced/max_size; original VGGT exposes crop/pad.
                default_mode = signature["mode"].default
                options["mode"] = default_mode if default_mode in ("balanced", "max_size", "crop") else "crop"
            images = self.preprocess(paths, **options).to(self.device)
            # Landscape vendor crop mode changes resolution only.
            ih, iw = images.shape[-2:]
            if abs(ih/iw-height/width) > .04:
                raise ValueError("unexpected vendor crop/pad; refusing invalid pixel geometry")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
                output = self.model(images)
            depth = output["depth"].float().cpu().numpy().reshape(len(rgb), ih, iw)
            confidence = output["depth_conf"].float().cpu().numpy().reshape(len(rgb), ih, iw)
        # Resize both values and validity support; never interpolate through NaN.
        result = []
        for d, c in zip(depth, confidence):
            known = np.isfinite(d) & (d > 0) & np.isfinite(c) & (c > 0)
            resized = cv2.resize(np.where(known, d, 0.), (width, height), interpolation=cv2.INTER_LINEAR)
            support = cv2.resize(known.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
            result.append(np.where(support >= 1.-1e-6, resized, 0.))
        return np.asarray(result, np.float32)


def load_backend(spec, config):
    """User-supplied module:factory for an existing mask/depth/tracker backend."""
    module, factory = spec.split(":", 1)
    return getattr(importlib.import_module(module), factory)(config)
