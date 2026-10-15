from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/run'))
from expand_dexycb_snapshot import remap_row_references


def test_nested_manual_and_motion_audit_rows_follow_replacement_mapping():
    value={'rows':[{'row':9674,'source_sequence':'episode','manual_decision':{'row':9674,'verdict':'exclude_tail'}}]}
    actual=remap_row_references(value,{9674:9574})
    row=actual['rows'][0]
    assert row['row']==9574 and row['parent_row']==9674
    assert row['manual_decision']['row']==9574 and row['manual_decision']['parent_row']==9674
    assert value['rows'][0]['row']==9674
