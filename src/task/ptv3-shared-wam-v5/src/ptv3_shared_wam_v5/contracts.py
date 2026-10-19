"""Versioned v5 boundaries, independent of v4 checkpoints and labels."""
SCHEMA = "ref2dex.ptv3-shared-wam.v5"
DATA_SCHEMA = "ref2dex.wam-v5-sequence-overlay.v1"
STATS_SCHEMA = "ref2dex.wam-v5-train-horizon-stats.v1"
HORIZON, HISTORY, HAND_POINTS = 24, 4, 21
SURFACE_POINTS, OBJECT_POINTS, FLOW_POINTS = 778, 2048, 32
PART_NAMES = ("palm",) + tuple(f"{finger}_{segment}" for finger in
    ("thumb", "index", "middle", "ring", "little") for segment in ("proximal", "middle", "distal"))
C_DIM = 1 + len(PART_NAMES)
# Root/base terminology follows each source annotation, particularly thumb CMC.
JOINT_NAMES = ("wrist",) + tuple(f"{finger}_{joint}" for finger in
    ("thumb", "index", "middle", "ring", "little") for joint in ("base", "middle", "distal", "tip"))
SPARSE11_IN_21 = (0, 1, 5, 9, 13, 17, 4, 8, 12, 16, 20)
MANO_JOINTS_IN_21 = (0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20)
MANO_JOINT_TO_PART = (0, 4, 5, 6, 7, 8, 9, 13, 14, 15, 10, 11, 12, 1, 2, 3)
INPUT_SEMANTICS = "v5.observed-joint21-surface778-object2048.h4.v1"
LABEL_SEMANTICS = "v5.joint21-delta.surface-vertex-proximity16.r3mm-falloff10mm.v1"
SAMPLING_SEMANTICS = "v5.inherited-main3-motion-strata-native-uniform.v1"
