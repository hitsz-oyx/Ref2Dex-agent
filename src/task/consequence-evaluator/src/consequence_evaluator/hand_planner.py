"""Prospective G -> frozen PW -> common frozen C1; no physical candidate fork."""
from pathlib import Path
import sys
import numpy as np
import torch

from .data import sha
from .hand_execution import SCHEMA,HandExecution,current_frame,compose_motion
from .old_utility import OldUtility,pw_sample,future_from_prediction,deterministic_group_mean

ROOT=Path(__file__).resolve().parents[5]
sys.path.insert(0,str(ROOT/'src/task/cm-interaction-oracle/src'))
from oracle_y_utility import candidate_deltas


class HandPlanner:
    def __init__(self, bridge, evaluator, pointworld, canonical):
        PW=ROOT/'src/task/cm-pointflow-effect-pretrain';sys.path.insert(0,str(PW/'src'))
        from oakink_wm.pointworld_temporal import model_from_config,capped_collate,VENDOR
        from oakink_wm.pointworld_performance import install_fused_hilbert
        import oakink_wm.pointworld as spatial
        import oakink_wm.pointworld_temporal as temporal
        self.bridge_path=Path(bridge);g=torch.load(bridge,map_location='cpu',weights_only=False)
        if g['schema']!=SCHEMA or g['arm']!='HA':raise ValueError('action-conditioned frozen bridge required')
        self.g=HandExecution(**g['architecture']).cuda().eval();self.g.load_state_dict(g['model'])
        self.gstats=tuple(x.cuda() for x in g['statistics']);self.gunit=g['output_unit_m'];self.plan_unit=g['plan_unit']
        q=torch.load(evaluator,map_location='cpu',weights_only=False)
        if q['arm']!='C1':raise ValueError('same frozen C1 evaluator required')
        self.q=OldUtility(**q['architecture']).cuda().eval();self.q.load_state_dict(q['model'])
        self.qstats={k:tuple(x.cuda() for x in v) for k,v in q['statistics'].items()}
        state=torch.load(pointworld,map_location='cpu',weights_only=False)
        self.source_hashes={}
        for base,key in ((PW,'implementation_sources'),(VENDOR,'vendor_sources')):
            for name,digest in state['identity'][key].items():
                path=(base/name).resolve()
                if sha(path)!=digest:raise ValueError('PW frozen source drift: '+str(path))
                self.source_hashes[str(path)]=digest
        install_fused_hilbert();spatial.mean_groups=temporal.mean_groups=deterministic_group_mean
        self.pw=model_from_config(state['identity']['stats'],state['config']).cuda().eval();self.pw.load_state_dict(state['model'])
        self.collate=capped_collate;del state
        with np.load(canonical,allow_pickle=False) as f:self.canonical={k:f[k] for k in f.files}
        self.plan=np.zeros((7,24,18),np.float32);self.plan[:,:8]=candidate_deltas().numpy()[:,None]
        self.hashes={str(Path(p).resolve()):sha(p) for p in (bridge,evaluator,pointworld,canonical)}
        self.hashes.update(self.source_hashes)
        self.first_repeat_checked=False

    @torch.inference_mode()
    def predict_hand(self, history, objects, hands, plans):
        state=np.concatenate((history,hands.reshape(len(hands),-1)),-1).astype('float32')
        mean,std=self.gstats;x=(torch.from_numpy(state).cuda()-mean)/std
        motion=self.g(x,torch.from_numpy(plans).cuda()/self.plan_unit)*self.gunit
        return compose_motion(torch.from_numpy(hands[:,-1]).cuda(),motion).cpu().numpy()

    @torch.inference_mode()
    def predict_object(self, objects,hands,future):
        values=[]
        for begin in range(0,len(future),8):
            end=min(begin+8,len(future))
            samples=[pw_sample(objects[i],hands[i],future[i],self.canonical,i) for i in range(begin,end)]
            batch={k:v.cuda() for k,v in self.collate(samples).items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=self.pw(batch,'action')
            if not self.first_repeat_checked:
                with torch.autocast('cuda',dtype=torch.bfloat16):repeat=self.pw(batch,'action')
                if not torch.equal(pred['rotation'],repeat['rotation']) or not torch.equal(pred['translation'],repeat['translation']):
                    raise ValueError('PW numerical repeatability failed')
                self.first_repeat_checked=True
            r=pred['rotation'][:,0].float().cpu().numpy();t=pred['translation'][:,0].float().cpu().numpy()
            values.extend(future_from_prediction(r[j],t[j],future[i]) for j,i in enumerate(range(begin,end)))
        return np.stack(values)

    @torch.inference_mode()
    def score(self,history,plans,future):
        inputs={k:torch.from_numpy(v).float().cuda() for k,v in dict(history=history,action=plans,future=future).items()}
        for k,v in inputs.items():mean,std=self.qstats[k];inputs[k]=(v-mean)/std
        scores=[]
        for begin in range(0,len(history),252):
            s=slice(begin,begin+252)
            scores.append(self.q(inputs['history'][s],inputs['action'][s],inputs['future'][s]).cpu().numpy())
        result=np.concatenate(scores)
        if not np.isfinite(result).all():raise ValueError('nonfinite candidate scores')
        return result

    def choose(self,history,world_objects,world_hands,rows):
        n=len(history);choices=np.zeros(n,int);scores=np.zeros((n,7),np.float32)
        objects=[];hands=[]
        for e in rows:
            o,h=current_frame(world_objects[:,e],world_hands[:,e]);objects.extend([o]*7);hands.extend([h]*7)
        objects=np.array(objects);hands=np.array(hands);raw=np.repeat(history[rows],7,axis=0)
        plans=np.tile(self.plan,(len(rows),1,1));predhand=self.predict_hand(raw,objects,hands,plans)
        future=self.predict_object(objects,hands,predhand);value=self.score(raw,plans,future).reshape(len(rows),7)
        choices[rows]=value.argmax(1);scores[rows]=value
        return choices,scores

    def verify(self):
        if any(sha(p)!=h for p,h in self.hashes.items()):raise ValueError('planner frozen input drift')
