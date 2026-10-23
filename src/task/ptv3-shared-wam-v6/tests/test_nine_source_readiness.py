"""Continuity audit must distinguish valid geometry from a crossed segment."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0,str(Path(__file__).parents[1]/'tools/audit'))
import audit_nine_source_readiness as census


def test_clock_mask_excludes_only_windows_crossing_native_segment(tmp_path):
    packet=tmp_path/'packet.npz'
    segment=np.zeros(80,np.int32);segment[32:]=1
    np.savez(packet,time=np.arange(80)/30,segment=segment,native_action_label=np.ones(80))
    tuples=np.array([[0,0,3],[0,0,20],[0,0,50]],np.int32)
    valid,evidence=census.clock_mask(dict(source_row=dict(source_path=str(packet))),
        dict(source_id=9,frames=80),tuples,{})
    assert valid.tolist()==[True,False,True]
    assert evidence['intent_label_fields']==['native_action_label']
    assert not evidence['complete_certificate_fields']
