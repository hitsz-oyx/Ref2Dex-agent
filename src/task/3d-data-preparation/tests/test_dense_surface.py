import json

import numpy as np
import pytest
import trimesh

from pose_mesh_data.dense_surface import SCHEMA, apply_surface_overlay, sample_mesh_surface


def plane():
    vertices = np.array([[x / 50, y / 50, 0.] for x in range(6) for y in range(6)])
    faces = []
    for x in range(5):
        for y in range(5):
            a = x * 6 + y
            faces.extend(((a, a+6, a+1), (a+1, a+6, a+7)))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    original = dict(surface=vertices.astype(np.float32),
                    normals=np.tile([0., 0., 1.], (36, 1)).astype(np.float32),
                    query=vertices[:32].astype(np.float32))
    return mesh, original


def test_dense_samples_lie_on_cad_and_retain_flow_identities():
    mesh, original = plane()
    a = sample_mesh_surface(mesh, original, "plane")
    b = sample_mesh_surface(mesh, original, "plane")
    assert len(np.unique(a["surface"], axis=0)) == 512
    assert (a["surface"][:, :2] >= 0).all() and (a["surface"][:, :2] <= .1).all()
    assert (a["surface"][:, 2] == 0).all()
    np.testing.assert_array_equal(a["surface"], b["surface"])
    np.testing.assert_array_equal(a["query"], original["query"])
    np.testing.assert_allclose(np.linalg.norm(a["normals"], axis=-1), 1.)
    with pytest.raises(ValueError, match="same frame"):
        sample_mesh_surface(mesh, original, "wrong-scale", scale=.01)


def test_overlay_preserves_source_and_rejects_changed_flow(tmp_path):
    mesh, original = plane()
    replacement = sample_mesh_surface(mesh, original, "plane")
    np.savez(tmp_path / "plane.npz", **replacement)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(dict(schema=SCHEMA, status="COMPLETED",
                                       objects=[dict(key="plane", path="plane.npz")])))
    geometry = dict(plane=original)
    apply_surface_overlay(geometry, manifest)
    assert geometry["plane"]["surface"].shape == (512, 3)
    assert original["surface"].shape == (36, 3)
    replacement["query"][0] += .01
    np.savez(tmp_path / "plane.npz", **replacement)
    with pytest.raises(ValueError, match="identity mismatch"):
        apply_surface_overlay(dict(plane=original), manifest)
