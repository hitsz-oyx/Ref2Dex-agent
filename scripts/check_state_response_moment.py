"""Math smoke: randomized moments remove drift; paired risk detects correct field."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.fit_state_response_field import paired_risk,cell_ids

def main():
    rng=np.random.RandomState(183);n=300000
    s=rng.choice([-1.,1.],size=n);eps=rng.randn(n,12)
    true=np.zeros((n,12,6));true[:,0,0]=s
    drift=(s*3+5)[:,None]*np.ones((1,6));y=drift+np.einsum('nak,na->nk',true,eps)
    moment=eps[:,:,None]*(y-drift)[:,None,:]
    assert abs(np.mean(moment[s>0,0,0])-1)<.02 and abs(np.mean(moment[s<0,0,0])+1)<.02
    risk=paired_risk(true.reshape(n,72),np.zeros((n,72)),moment.reshape(n,72)).mean()
    assert abs(risk+1/72)<.0005
    assert np.allclose(paired_risk(true.reshape(n,72),true.reshape(n,72),moment.reshape(n,72)),0)
    assert np.array_equal(cell_ids(np.array([9,10,25,26,100,101]),np.zeros(6,int),10,100),[0,1,1,2,2,3])
    source=(ROOT/'scripts/audit_continuous_critic_panel.py').read_text()
    for marker in ["a.panel not in list(range(547,567))+[568,569]","m['experiment_id']!='P-20261002-continuous-critic-policy'","r['deterministic']!=(a.panel in (568,569))"]:assert source.count(marker)==1
    print('PASS: conditional moment sign, drift subtraction, paired risk normalization, zero matched difference, fixed phase boundaries, narrow audit wrapper')

if __name__=='__main__':main()
