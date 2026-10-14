"""Current-only Grounding DINO/SAM2 selection and offline identity masks.

Automatic outputs stay provisional until explicit visual review. Future RGB
enters prepare_labels only after current selection and queries are finalized.
"""
import hashlib
import inspect
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np

from .geometry import transform_points
from .io import atomic_json


def target_phrase(description):
    text = description.lower().strip()
    # Source/destination containers must never replace an unsupported object.
    prefix = re.split(r"\b(?:from|off|with|out of|into|onto|inside|and (?:put|place)|then (?:put|place))\b", text)[0]
    if re.search(r"\b(?:cloth|towel|sponge|scrub|transparent|translucent|clear|glass|plastic bag|paper|food)\b", prefix):
        raise ValueError("TARGET_UNSUPPORTED_MATERIAL")
    heads = r"lego(?: block| brick)?|mustard bottle|bottle|cup|mug|box|block|brick|can|bowl|container|toy"
    matches = list(re.finditer(r"\b(?:white|black|red|blue|green|yellow|orange|pink|purple|small|large|wooden|metal)?\s*(?:"+heads+r")\b", prefix))
    if len(matches) != 1:
        raise ValueError("TARGET_LANGUAGE_AMBIGUOUS")
    # Preserve shape/material/color modifiers. Dropping "yellow rhombus" from
    # "yellow rhombus block" made the generic grounder choose a drying rack.
    noun = re.sub(r"^pick\s+up\s+(?:(?:a|an|the)\s+)?", "", prefix).strip(" .,")
    if (not prefix.startswith("pick up ") or len(noun.split()) > 8
            or matches[0].end() != len(prefix.rstrip(" .,"))):
        raise ValueError("TARGET_LANGUAGE_UNSUPPORTED_PARSE")
    return noun.replace("lego", "lego toy")


def box_iou(a, b):
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    inter = float(np.maximum(hi-lo, 0).prod())
    return inter/max(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2]))-inter, 1.)


def choose_box(boxes, scores, hand_uv, shape):
    """Refuse separated near-ties rather than choose using future motion."""
    h, w = shape
    candidates = []
    for box, score in zip(boxes, scores):
        box = np.clip(np.asarray(box, np.float32), [0, 0, 0, 0], [w-1, h-1, w-1, h-1])
        area = float(np.maximum(box[2:]-box[:2], 0).prod())
        if not 64 <= area <= .35*h*w:
            continue
        closest = np.maximum(np.maximum(box[:2]-hand_uv, hand_uv-box[2:]), 0)
        distance = float(np.linalg.norm(closest, axis=-1).min()) if len(hand_uv) else float("inf")
        if distance <= 180:
            candidates.append(dict(box=box, score=float(score), rank=float(score)-.5*distance/90.,
                                   hand_distance_px=distance))
    candidates.sort(key=lambda x: x["rank"], reverse=True)
    if not candidates:
        raise ValueError("TARGET_NOT_GROUNDED_NEAR_OBSERVED_HAND")
    best = candidates[0]
    if any(box_iou(best["box"], other["box"]) < .5 and best["rank"]-other["rank"] < .05
           for other in candidates[1:]):
        raise ValueError("TARGET_CURRENT_DETECTIONS_AMBIGUOUS")
    return best


def color_evidence(rgb, visible, phrase):
    """Reject clear named-color contradictions; do not infer white balance.

    Warm illumination of white plastic is allowed. This small consistency
    screen cannot establish semantic identity and does not remove visual review.
    """
    pixels = np.asarray(rgb, np.float32)[visible]/255.
    if len(pixels) < 64:
        raise ValueError("TARGET_COLOR_SUPPORT_INSUFFICIENT")
    r, g, b = pixels.T
    color = phrase.split()[0]
    green = (g > .99*r) & (g > 1.25*b)
    blue = (b > 1.1*r) & (b > 1.02*g)
    red = (r > 1.6*g) & (r > 1.6*b)
    conflict = np.zeros(len(pixels), bool)
    if color == "white":
        conflict = green | blue | red
    elif color == "blue":
        conflict = green | red
    elif color == "green":
        conflict = blue | red
    fraction = float(conflict.mean())
    if fraction > .5:
        raise ValueError("TARGET_NAMED_COLOR_CONTRADICTION")
    return dict(named_color=color, contradictory_fraction=fraction, median_rgb=np.median(pixels, axis=0).tolist())


class GroundedSAMMasks:
    def __init__(self, config):
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.config = config["automatic_masks"]
        local = str(Path(self.config["grounding_model"]).resolve())
        self.processor = AutoProcessor.from_pretrained(local, local_files_only=True)
        self.grounder = AutoModelForZeroShotObjectDetection.from_pretrained(local,
            local_files_only=True).eval().to("cuda:0")
        sys.path.insert(0, str(Path(self.config["sam_vendor"]).resolve()))
        from sam2.build_sam import build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.video_model = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml",
            self.config["sam_checkpoint"], device="cuda:0", apply_postprocessing=False)
        self.image_model = SAM2ImagePredictor(self.video_model)
        self.scratch = Path(self.config["scratch"]).resolve()
        self.audit_root = Path(self.config["audit_root"]).resolve()
        for root in (self.scratch, self.audit_root):
            if root == Path("/tmp") or Path("/tmp") in root.parents:
                raise ValueError("owned project mask paths required")
            root.mkdir(parents=True, exist_ok=True)
        self.active, self.future = None, None

    def _detect(self, rgb, text):
        import torch
        from PIL import Image
        inputs = self.processor(images=Image.fromarray(rgb), text=text+".", return_tensors="pt").to("cuda:0")
        with torch.inference_mode():
            outputs = self.grounder(**inputs)
        post = self.processor.post_process_grounded_object_detection
        threshold = "threshold" if "threshold" in inspect.signature(post).parameters else "box_threshold"
        result = post(outputs, inputs.input_ids, **{threshold: .25}, text_threshold=.2,
                      target_sizes=[rgb.shape[:2]])[0]
        return result["boxes"].float().cpu().numpy(), result["scores"].float().cpu().numpy()

    def _mask(self, box):
        masks, scores, _ = self.image_model.predict(box=np.asarray(box), multimask_output=False)
        if not len(scores) or float(scores[0]) < .6:
            raise ValueError("TARGET_SAM_LOW_MASK_QUALITY")
        return np.asarray(masks[0], bool), float(scores[0])

    def select_current(self, key, tick, rgb, native):
        import cv2
        import torch
        began = time.monotonic()
        self.active, self.future = None, None
        phrase = target_phrase(native.language)
        points = transform_points(native.hand[-1], np.linalg.inv(native.camera[-1]))
        projected = points @ native.intrinsic.T
        good = native.valid[-1] & (points[..., 2] > 0)
        motion = []
        for side in range(2):
            supported = native.valid[:, side].all(0)
            values = np.linalg.norm(native.hand[-1, side]-native.hand[0, side], axis=-1)
            motion.append(float(np.median(values[supported])) if supported.sum() >= 3 else 0.)
        active = int(np.argmax(motion))
        # A clear already-observed interacting hand is a current-only prior.
        # Weak/equal motion leaves both hands and ambiguity is rejected below.
        if motion[active] >= .002 and motion[active] >= 2*(motion[1-active]+.0001):
            good[1-active] = False
        else:
            active = None
        uv = projected[good, :2]/projected[good, 2:]
        boxes, scores = self._detect(rgb, phrase)
        choice = choose_box(boxes, scores, uv, rgb.shape[:2])
        choice.update(observed_hand_motion_m=motion, observed_active_hand=active)
        hand_boxes, hand_scores = self._detect(rgb, "human hand . human forearm")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.image_model.set_image(rgb)
            target, mask_score = self._mask(choice["box"])
            hand = np.zeros(rgb.shape[:2], bool)
            for box in hand_boxes[np.argsort(-hand_scores)[:4]]:
                try:
                    part, _ = self._mask(box)
                    if part.sum() < .5*part.size:
                        hand |= part
                except ValueError:
                    continue
        if not hand.any():
            raise ValueError("HAND_SEGMENTATION_UNAVAILABLE")
        hand = cv2.dilate(hand.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        visible = target & ~hand
        if visible.sum() < 64 or visible.sum() < .2*target.sum() or target.sum() > .25*target.size:
            raise ValueError("TARGET_MASK_OCCLUDED_OR_IMPURE")
        color = color_evidence(rgb, visible, phrase)
        folder = self.audit_root/(hashlib.sha256(key.encode()).hexdigest()[:16]+"-"+str(tick))
        folder.mkdir(parents=True, exist_ok=True)
        choice["box"] = choice["box"].tolist()
        evidence = dict(key=key, tick=tick, target=phrase, description=native.language, detection=choice,
            color_consistency=color,
            mask_score=mask_score, visible_pixels=int(visible.sum()), verification="automatic-provisional",
            current_only=True, selection_s=time.monotonic()-began)
        atomic_json(folder/"selection.json", evidence)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        image[visible] = (.55*image[visible]+.45*np.array([0, 255, 0])).astype(np.uint8)
        image[hand] = (.65*image[hand]+.35*np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(str(folder/"current.jpg"), image)
        cv2.imwrite(str(folder/"target.png"), target.astype(np.uint8)*255)
        cv2.imwrite(str(folder/"hand.png"), hand.astype(np.uint8)*255)
        self.active = dict(key=key, tick=tick, target=target.copy(), hand=hand.copy(), folder=folder,
                           evidence=evidence)
        return dict(target=target, hand=hand, name=phrase, current_verified=True, training_allowed=False,
                    verification="automatic-provisional", method="current-grounding-dino-sam2-provisional",
                    evidence=evidence)

    def prepare_labels(self, key, frame_ids, rgb):
        import cv2
        import torch
        if (self.active is None or self.active["key"] != key or len(frame_ids) != 25
                or int(frame_ids[0]) != self.active["tick"]):
            raise ValueError("future mask request differs from current identity")
        h, w = rgb.shape[1:3]
        result, overlay = np.zeros((25, h, w), bool), []
        with tempfile.TemporaryDirectory(dir=self.scratch, prefix="sam-label-") as folder:
            for i, frame in enumerate(rgb):
                if not cv2.imwrite(str(Path(folder)/("%05d.jpg" % i)), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise IOError("mask label scratch write failed")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                state = self.video_model.init_state(folder, offload_video_to_cpu=True, offload_state_to_cpu=True)
                self.video_model.add_new_mask(state, frame_idx=0, obj_id=1, mask=self.active["target"])
                self.video_model.add_new_mask(state, frame_idx=0, obj_id=2, mask=self.active["hand"])
                for step, ids, logits in self.video_model.propagate_in_video(state,
                        start_frame_idx=0, max_frame_num_to_track=25):
                    masks = (logits[:, 0] > 0).cpu().numpy()
                    obj, hand = masks[list(ids).index(1)], masks[list(ids).index(2)]
                    hand = cv2.dilate(hand.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                    result[step] = obj & ~hand
                    if step in (0, 8, 16, 24):
                        image = cv2.cvtColor(rgb[step], cv2.COLOR_RGB2BGR)
                        image[result[step]] = (.55*image[result[step]]+.45*np.array([0, 255, 0])).astype(np.uint8)
                        cv2.putText(image, "t+%d %s" % (step, self.active["evidence"]["target"]), (12, 24),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                        overlay.append(image)
                self.video_model.reset_state(state)
                del state
        result[0] = self.active["target"] & ~self.active["hand"]
        if len(overlay) == 4:
            sheet = np.concatenate([np.concatenate(overlay[:2], 1), np.concatenate(overlay[2:], 1)], 0)
            cv2.imwrite(str(self.active["folder"]/"future.jpg"), sheet)
        self.future = result

    def label_masks(self, key, frame_ids, shape):
        if (self.active is None or key != self.active["key"] or int(frame_ids[0]) != self.active["tick"]
                or self.future is None or self.future.shape != (len(frame_ids), *shape)):
            return None
        return self.future.copy()


def create(config):
    return GroundedSAMMasks(config)
