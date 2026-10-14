"""Independent physical forecast and first-control checkpoint criteria."""
import math
from pathlib import Path

SELECTION_ALIASES = ("best.pt", "latest.pt", "forecast_best.pt", "control_best.pt")


def selection_keys(report, stage):
    if stage != "robot":
        return {}
    metrics = report["metrics"]
    groups = []
    for suffix in ("/hand_all24", "/scene_all24"):
        rows = [v for k, v in metrics.items() if k.startswith("source") and k.endswith(suffix)]
        count = sum(v["count"] for v in rows)
        if not count:
            raise ValueError("physical forecast selection requires nonempty hand/scene metrics")
        groups.append(sum(v["mean_mm"] * v["count"] for v in rows if v["count"]) / count)
    keys = dict(forecast_best=(sum(groups),),
                control_best=tuple(metrics[k]["p95"] for k in
                    ("GT/PD_xyz_mm", "pred/PD_xyz_mm", "GT/PD_rotation_rad", "GT/PD_finger_rad")))
    if not all(math.isfinite(x) for values in keys.values() for x in values):
        raise ValueError("finite checkpoint selection metrics required")
    return keys


class SelectionTracker:
    """Historical scores count only when their immutable weights still exist."""
    def __init__(self, output, stage, curve=()):
        self.output, self.stage = Path(output), stage
        self.best = {}
        self.missing_steps = []
        for row in curve:
            if not (self.output / ("step%d.pt" % row["step"])).is_file():
                self.missing_steps.append(row["step"])
                continue
            self.update(row, row["step"])

    def update(self, report, step):
        changed = []
        for name, key in selection_keys(report, self.stage).items():
            if name not in self.best or key < self.best[name]["key"]:
                self.best[name] = dict(key=key, step=step)
                changed.append(name)
        return changed

    def report(self):
        return dict(criteria=dict(forecast_best="hand_all24_mm + scene_all24_mm; point weighted within each group",
                                  control_best="GT first xyz p95, predicted first xyz p95, GT rotation p95, GT finger p95"),
                    best=self.best, historical_weights_missing=self.missing_steps)
