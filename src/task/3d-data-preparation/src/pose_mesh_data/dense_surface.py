"""Derived CAD surface samples, retaining the original 32 flow identities."""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "ref2dex.wam-v4-dense-surface.v1"


def sample_mesh_surface(mesh, original, key, scale=1.):
    triangles = np.asarray(mesh.triangles, np.float64) * scale
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    area = np.linalg.norm(cross, axis=-1)
    valid = np.isfinite(triangles).all(axis=(1, 2)) & (area > 1e-15)
    triangles, cross, area = triangles[valid], cross[valid], area[valid]
    if not len(triangles):
        raise ValueError("mesh has no finite nondegenerate triangles: " + key)
    query = np.asarray(original["query"], np.float32)
    if query.shape != (32, 3) or len(np.unique(query, axis=0)) != 32:
        raise ValueError("32 distinct original flow queries required")
    # The old geometry was sampled from CAD vertices. Check the common frame
    # and scale explicitly before reusing it with newly sampled surface points.
    vertices = set(map(tuple, np.round(triangles.reshape(-1, 3), 6)))
    if any(tuple(point) not in vertices for point in np.round(query.astype(np.float64), 6)):
        # Float32 conversion near a rounding boundary can differ by one bin;
        # coordinate-wise equality within 1um remains a direct vertex check.
        flat = triangles.reshape(-1, 3)
        if any(not np.isclose(flat, point, atol=1e-6, rtol=0).all(-1).any() for point in query):
            raise ValueError("original query is not a CAD vertex in the same frame: " + key)
    old = np.asarray(original["surface"])
    query_normals = []
    for point in query:
        matches = np.flatnonzero(np.isclose(old, point, atol=1e-8, rtol=0).all(-1))
        if not len(matches):
            raise ValueError("original query has no preserved normal: " + key)
        query_normals.append(original["normals"][matches[0]])
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))
    face = rng.choice(len(triangles), 480, p=area / area.sum())
    bary = rng.random((480, 2))
    bary[bary.sum(-1) > 1] = 1 - bary[bary.sum(-1) > 1]
    chosen = triangles[face]
    sampled = (chosen[:, 0] + bary[:, :1] * (chosen[:, 1] - chosen[:, 0])
               + bary[:, 1:] * (chosen[:, 2] - chosen[:, 0]))
    surface = np.concatenate((query, sampled)).astype(np.float32)
    normals = np.concatenate((query_normals, cross[face] / area[face, None])).astype(np.float32)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    if not np.isfinite(normals).all() or (lengths <= 1e-6).any():
        raise ValueError("nonfinite or zero CAD normals: " + key)
    normals /= lengths
    if len(np.unique(surface, axis=0)) != 512:
        raise ValueError("dense CAD sampling did not produce 512 distinct points: " + key)
    return dict(surface=surface, normals=normals, query=query.copy())


def apply_surface_overlay(geometry, path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "COMPLETED":
        raise ValueError("completed v4 dense surface manifest required")
    seen = set()
    for row in manifest["objects"]:
        key = row["key"]
        if key not in geometry or key in seen:
            raise ValueError("unknown or duplicated dense surface identity: " + key)
        with np.load(path.parent / row["path"], allow_pickle=False) as values:
            replacement = {key: values[key].copy() for key in values.files}
        if (replacement["surface"].shape != (512, 3)
                or len(np.unique(replacement["surface"], axis=0)) != 512
                or replacement["normals"].shape != (512, 3)
                or not np.isfinite(replacement["surface"]).all()
                or not np.isfinite(replacement["normals"]).all()
                or not np.allclose(np.linalg.norm(replacement["normals"], axis=-1), 1., atol=1e-5)
                or not np.array_equal(replacement["query"], geometry[key]["query"])):
            raise ValueError("dense surface/normal/flow identity mismatch: " + key)
        geometry[key] = replacement
        seen.add(key)
