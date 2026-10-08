"""Train-source empirical control-error bank and geometric pre-action stages.

Two failed examples define a pilot candidate bank, not a fitted failure law or
learner/expert covariance. Replaying policy deviations does not establish their
causal role in the observed failures.
"""
import numpy as np

from .collection import PHASES
from .contracts import K
from .value_outcomes import PARAMETERS

VALUE_PHASES = (*PHASES,'place')
BANK_SCHEMA = 'ref2dex.consequence-value.control-error-bank.v1'
COUPLED = [7,9,11,13,16,17]


def error_chunk(errors, gain=1.,residual_bound=.2):
    errors=np.asarray(errors)
    if (errors.ndim!=2 or errors.shape[1]!=18 or len(errors)<4
            or not np.isfinite(errors).all() or not 0<gain<=4. or not .2<=residual_bound<=.5):
        raise ValueError('finite pre-failure18control errors and bounded gain required')
    knots=np.stack([np.median(part,axis=0) for part in np.array_split(errors,4)])
    positions=np.linspace(0,1,K)
    result=np.stack([np.interp(positions,np.linspace(0,1,4),knots[:,j]) for j in range(18)],-1)
    result *= gain*np.sin(np.pi*positions)[:,None]**2
    result[:,:3]=np.clip(result[:,:3],-.05,.05)
    result=np.clip(result,-residual_bound,residual_bound);result[:,COUPLED]=0;result[[0,-1]]=0
    return result.astype('float32')


def sample_bank(bank,count,seed):
    if (bank.get('schema')!=BANK_SCHEMA or bank.get('source_split')!='train'
            or bank.get('phase') not in ('place','hold','contact')):
        raise ValueError('train-derived task-stage candidate bank required')
    chunks=np.asarray(bank['chunks'],dtype='float32')
    bound=bank.get('limits',{}).get('residual',.2)
    if not .2<=bound<=.5:raise ValueError('bank residual bound outside registered calibration range')
    if (chunks.ndim!=3 or chunks.shape[1:]!=(K,18) or not len(chunks)
            or not np.isfinite(chunks).all() or np.abs(chunks).max()>bound+1e-6
            or np.abs(chunks[:,:,:3]).max()>.05+1e-6):
        raise ValueError('invalid empirical chunk bank')
    if np.any(chunks[:,:,COUPLED]) or np.any(chunks[:,[0,-1]]):
        raise ValueError('bank changes coupled channels or zero boundaries')
    ids=np.random.default_rng(seed).integers(len(chunks),size=count)
    return chunks[ids].copy(),ids


class ValuePhaseTracker:
    """No force proxy; placing follows reference boundary after a prior45hold."""
    def __init__(self,place_start):
        self.place_start=np.asarray(place_start,dtype='int64')
        self.near_run=np.zeros(len(self.place_start),np.int64)
        self.held_run=self.near_run.copy();self.completed=np.zeros(len(self.place_start),bool)

    def measure(self,height,initial_height,gap,support,footprint,tick,active):
        valid=np.asarray(active,dtype=bool)&(tick>=1)
        near=valid&(np.asarray(gap)<=PARAMETERS['surface_gap_m'])
        supported=valid&np.asarray(footprint,dtype=bool)&(np.abs(support)<=PARAMETERS['support_gap_m'])
        held=near&((np.asarray(height)-initial_height)>=PARAMETERS['lift_m'])&~supported
        self.near_run=np.where(near,self.near_run+1,0)
        self.held_run=np.where(held,self.held_run+1,0)
        self.completed |= self.held_run>=PARAMETERS['stable_frames']
        code=np.zeros(len(valid),np.int64);code[near]=1;code[self.near_run>=3]=2
        code[held]=3;code[self.held_run>=PARAMETERS['stable_frames']]=4
        code[(tick>=self.place_start)&self.completed&near]=5
        return code
