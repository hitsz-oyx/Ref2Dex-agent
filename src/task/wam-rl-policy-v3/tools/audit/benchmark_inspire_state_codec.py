"""CPU-only engineering timing of the lossless native state wire format."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np
import torch

TASK=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TASK/'src'))
from wam_rl_policy_v3.inspire_state_codec import pack_state,unpack_state,FIELDS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('fresh timing output required')
    f=torch.load(a.fixture,map_location='cpu',weights_only=False);x=f['current'][:256].numpy()
    state=dict(q=x[:,:18],dq=x[:,18:36],physical=x[:,36:284],objects=x[:,284:297],
        hand=f['batch']['start_hand'][:256,0].numpy(),tick=np.arange(256,dtype=np.int64),
        episodes=[7]*256,sequences=['s10_apple_eat_1']*256)
    result={}
    for mode in ('json_lists','packed_base64'):
        times=[]
        for _ in range(30):
            start=time.perf_counter()
            packet=(pack_state(state) if mode=='packed_base64' else
                    {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in state.items()})
            payload=json.dumps(packet,allow_nan=False,separators=(',',':')).encode()
            decoded=json.loads(payload)
            decoded=(unpack_state(decoded) if mode=='packed_base64' else
                     {k:(np.asarray(v,dtype=state[k].dtype) if k in FIELDS else v) for k,v in decoded.items()})
            times.append(time.perf_counter()-start)
        for k in FIELDS:np.testing.assert_array_equal(decoded[k],state[k])
        result[mode]=dict(median_s=statistics.median(times),bytes=len(payload),roundtrip_bitwise_equal=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
