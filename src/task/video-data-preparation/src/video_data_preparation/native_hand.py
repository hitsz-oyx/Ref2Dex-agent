"""Native 2x11 semantic hand points and continuous 30Hz windows."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .geometry import check_transform, transform_points

JOINT_SUFFIXES = ("Hand", "ThumbKnuckle", "IndexFingerKnuckle", "MiddleFingerKnuckle",
                  "RingFingerKnuckle", "LittleFingerKnuckle", "ThumbTip", "IndexFingerTip",
                  "MiddleFingerTip", "RingFingerTip", "LittleFingerTip")


@dataclass
class NativeEpisode:
    hand: np.ndarray
    valid: np.ndarray
    camera: np.ndarray
    intrinsic: np.ndarray
    session: str
    language: str

    @property
    def frames(self):
        return len(self.camera)

    def window(self, tick, anchor):
        if not 3 <= tick < self.frames - 24:
            raise ValueError("contiguous H4/K24 window outside episode")
        h = transform_points(self.hand[tick-3:tick+25], anchor)
        valid = self.valid[tick-3:tick+25]
        present = valid[:4].all(0)
        start = np.where(present[..., None], h[3], 0.)
        dynamics = np.concatenate([(h[3]-h[2])*30., (h[3]-2*h[2]+h[1])*900., h[3]-h[0]], -1)
        future_valid = valid[4:] & present[None]
        return dict(start_hand=start.astype(np.float32), start_hand_valid=present,
                    start_hand_history=np.where(present[..., None], dynamics, 0.).astype(np.float32),
                    trajectory_target=np.where(future_valid[..., None], h[4:]-h[3], 0.).astype(np.float32),
                    trajectory_valid=future_valid)


def read_native(path, confidence=.5):
    import h5py
    if not 0 < confidence <= 1:
        raise ValueError("confidence threshold must be in (0,1]")
    with h5py.File(path, "r") as f:
        camera = check_transform(f["transforms/camera"][:])
        if camera.ndim != 3 or len(camera) < 28:
            raise ValueError("episode shorter than H4+K24")
        if "confidences" not in f:
            raise ValueError("missing native confidences; never impute")
        hand = np.zeros((len(camera), 2, 11, 3), np.float32)
        valid = np.zeros(hand.shape[:-1], bool)
        for side_id, side in enumerate(("right", "left")):
            for joint, suffix in enumerate(JOINT_SUFFIXES):
                name = side + suffix
                if name not in f["transforms"] or name not in f["confidences"]:
                    continue
                transforms, scores = f["transforms"][name][:], f["confidences"][name][:]
                if transforms.shape != (len(camera), 4, 4) or scores.shape != (len(camera),):
                    raise ValueError("native hand clock mismatch")
                if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
                    raise ValueError("invalid native joint confidences")
                keep = np.isfinite(transforms).all((1, 2)) & (scores >= confidence)
                hand[:, side_id, joint] = np.where(keep[:, None], transforms[:, :3, 3], 0.)
                valid[:, side_id, joint] = keep
        k = np.asarray(f["camera/intrinsic"][:], np.float32)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
            raise ValueError("invalid native intrinsic")
        def attr(key, default=""):
            v = f.attrs.get(key, default)
            return v.decode() if isinstance(v, bytes) else str(v)
        key = "llm_description2" if attr("which_llm_description") == "2" else "llm_description"
        return NativeEpisode(hand, valid, camera, k,
                             attr("session_name", str(Path(path).resolve())), attr(key))
