"""Current robot-context policy takeovers, split by entire exploration prefix."""
import json
from pathlib import Path
import numpy as np
from .wam_effect_data import EffectData
from .wam_control_data import ControlData


def collection_entries(paths):
    entries=[]
    for path in paths:
        m=json.loads((Path(path)/'manifest.json').read_text())
        if m.get('experiment_id')!='P-20261013-goal-point-wam-visited-realign':raise ValueError('wrong collection experiment')
        entries.extend(m['admissions'])
    if len({(e['family'],e['takeover']) for e in entries})!=len(entries):raise ValueError('duplicate prefix/takeover')
    return sorted(entries,key=lambda e:(e['family'],e['takeover']))


class VisitedControlData(ControlData):
    def __init__(self,root,main,entries):
        EffectData.__init__(self,root,main)
        self.paths={'nominal':self.paths['nominal']};self.data={'nominal':self.data['nominal']}
        self.takeovers={};self.valid_ticks={};self.families={};self.excluded=[]
        expected=Path(root)/'outputs/goal-point-wam/wam-control-context-20261013-r3/stage2_step4500.pt'
        for i,entry in enumerate(entries):
            path=Path(entry['native']);m=json.loads((path/'manifest.json').read_text());audit=json.loads((path/'root_recovery_audit.json').read_text())
            if m['status']!='COMPLETED' or not audit['pass'] or m['controls']!=128 or not m.get('own_row_recovery'):raise ValueError('completed audited128 recovery required')
            if Path(m.get('control_identity',{}).get('checkpoint','')).resolve()!=expected.resolve():raise ValueError('current adapter policy required')
            family,takeover=entry['family'],int(m['takeover_tick'])
            shift=((0,0,0),(.5,0,0),(-.5,0,0),(0,.5,0))[family]
            if list(shift)!=m['prefix_exploration']['xyz_mm'] or takeover!=entry['takeover']:raise ValueError('exploration prefix differs')
            with np.load(path/'trajectory.npz',allow_pickle=False) as f:d={k:f[k].copy() for k in f.files}
            if not np.array_equal(d['action'],d['applied']) or not np.array_equal(d['applied'][takeover:,3],d['expert_action'][takeover:,1]):raise ValueError('own applied expert commands required')
            valid=[t for t in range(takeover,105) if d['expert_applied'][t:t+24,1].all() and not d['expert_clipped'][t:t+24,1].any()]
            if takeover not in valid or entry['held']<45:
                self.excluded.append(dict(entry,reason='clipped exact future or expert did not hold45'));continue
            name='recovery%d'%i
            self.paths[name]=path;self.data[name]=d;self.takeovers[name]=takeover
            self.valid_ticks[name]=valid;self.families[name]=family
        if not self.takeovers:raise ValueError('no recoverable current own-state expert future')

    def __getitem__(self,key):
        sample=super().__getitem__(key)
        if key[0].startswith('recovery'):sample['source_id']=np.int64(4+int(key[0][8:]))
        return sample
