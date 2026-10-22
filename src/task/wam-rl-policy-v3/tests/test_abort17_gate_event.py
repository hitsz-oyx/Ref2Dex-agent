import importlib.util
from pathlib import Path

def load():
    path=Path(__file__).resolve().parents[1]/'tools/audit/audit_v13_abort17.py';spec=importlib.util.spec_from_file_location('abort17',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_fallen_object_below_finite_slab_is_not_called_tabletop_penetration():
    m=load()
    assert m.table_gate_event([-.6,-.7],[False,True],-.55,.0055)=='fully_below_slab_footprint_reentry'
    assert m.table_gate_event([-.025,-.03],[False,True],.01,.0055)=='below_plane_footprint_reentry'

def test_actual_within_footprint_threshold_crossing_is_kept_even_for_large_step():
    m=load()
    assert m.table_gate_event([-.019,-.022],[True,True],.04,.0055)=='in_footprint_threshold_crossing'
    assert m.table_gate_event([.01,-.7],[True,True],-.55,.0055)=='in_footprint_threshold_crossing'
