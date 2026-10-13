"""Bounded sequential MP4 decode with native frame identity checks."""
from collections import OrderedDict
import json
import subprocess
import numpy as np


class VideoFrames:
    def __init__(self, path, native_frames, width=640, height=360):
        import cv2
        info = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=avg_frame_rate,nb_frames,width,height", "-of", "json", str(path)], text=True))["streams"][0]
        numerator, denominator = map(int, info["avg_frame_rate"].split("/"))
        if abs(numerator/denominator-30) > 1e-4 or int(info.get("nb_frames", -1)) != native_frames:
            raise ValueError("MP4/HDF5 native 30Hz clock mismatch")
        self.original_wh = (info["width"], info["height"])
        self.size = (width, height)
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            raise OSError("cannot open native MP4")
        self.next_frame, self.cache = 0, OrderedDict()

    def get(self, frame_ids):
        import cv2
        frame_ids = list(map(int, frame_ids))
        if not frame_ids or frame_ids != sorted(set(frame_ids)):
            raise ValueError("increasing unique native frame IDs required")
        result = []
        for frame in frame_ids:
            if frame < self.next_frame and frame not in self.cache:
                raise ValueError("decode requested expired frame; process anchors in order")
            while self.next_frame <= frame:
                success, bgr = self.capture.read()
                if not success:
                    raise ValueError("video ended before native frame " + str(self.next_frame))
                if self.next_frame >= frame_ids[0]:
                    rgb = cv2.cvtColor(cv2.resize(bgr, self.size), cv2.COLOR_BGR2RGB)
                    self.cache[self.next_frame] = rgb
                    while len(self.cache) > 28:
                        self.cache.popitem(last=False)
                self.next_frame += 1
            result.append(self.cache[frame])
        return np.stack(result)

    def close(self):
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
