"""A staged native URDF must have its actual meshes before a GPU launch."""
import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1]/'tools/run/stage_corrected_experts.py'
spec = importlib.util.spec_from_file_location('stage_corrected_experts_test', TOOL)
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def assets_fixture(folder, scale='1 1 1'):
    assets = folder/'assets'
    (assets/'mjcf').mkdir(parents=True)
    (assets/'inspire_hand_new').mkdir()
    (assets/'inspire_hand_new/inspire_hand_right.urdf').write_text('<robot/>')
    (assets/'mjcf/table.urdf').write_text('<robot/>')
    element = ('<origin xyz="0 0 0" rpy="0 0 0"/><geometry>'
               '<mesh filename="objects/duck/duck.obj" scale="'+scale+'"/></geometry>')
    (assets/'mjcf/duck.urdf').write_text('<robot><link><visual>'+element+'</visual>'
                                       '<collision>'+element+'</collision></link></robot>')
    return assets


def test_missing_native_mesh_blocks_readiness_even_when_urdf_exists(tmp_path):
    assets = assets_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match='native URDF mesh is missing'):
        stage.mesh_readiness(assets, ['duck'])


def test_canonical_copy_has_finite_dimensions_and_no_frame_transform(tmp_path):
    import trimesh
    assets = assets_fixture(tmp_path)
    mesh = assets/'mjcf/objects/duck/duck.obj'
    mesh.parent.mkdir(parents=True)
    trimesh.creation.box(extents=[.1, .2, .3]).export(mesh)
    measured = stage.mesh_readiness(assets, ['duck'])
    assert measured['duck']['dimensions_m'] == pytest.approx([.1, .2, .3])
    changed = (assets/'mjcf/duck.urdf').read_text().replace('scale="1 1 1"', 'scale="2 2 2"')
    (assets/'mjcf/duck.urdf').write_text(changed)
    with pytest.raises(ValueError, match='untransformed canonical mesh'):
        stage.mesh_readiness(assets, ['duck'])
