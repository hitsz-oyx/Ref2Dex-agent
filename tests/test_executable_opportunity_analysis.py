import numpy as np
from scripts.analyze_executable_contact_opportunity import compare


def balanced_randomization_fixture():
    pattern=[6,4,0,6,4,1,6,4,2,6,4,3,6,4,5,6,4,6,4,6,4,6,4,6,4]
    chosen=np.tile(pattern,80);n=len(chosen)
    return dict(assignment=chosen,propensity=np.where((chosen==4)|(chosen==6),.4,.04),
                episode=np.repeat(np.arange(40),50),group=np.repeat(np.arange(20),100),
                score=np.where(chosen==6,5.,np.where(chosen==4,-5.,0.)),risk=(chosen==4).astype(float),contact=np.ones(n))


def test_actual_base_propensity_recovers_known_effect_without_double_counting_duplicate_slots():
    d=balanced_randomization_fixture();mask=np.ones(len(d['assignment']),dtype=bool)
    r=compare(6,d,mask,2.,True)
    assert r['score_uplift_mm']==10. and r['label']=='PROMISING'
    assert r['lost_clearance_risk_difference']==-1.
    wrong={**d,'propensity':np.where(d['assignment']==4,.2,d['propensity'])}
    assert compare(6,wrong,mask,2.,True)['score_uplift_mm']==15.


def test_missing_matches_never_become_supported_opportunity():
    d=balanced_randomization_fixture();d['assignment'][d['assignment']==0]=1
    result=compare(0,d,np.ones(len(d['assignment']),dtype=bool),2.,True)
    assert result['support']['0']['matches']==0 and result['label']=='UNCLEAR'
    assert not result['gate']['passed']
