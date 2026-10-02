"""Keep full native/mesh audits; independently replace controller replay and gates."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    source=(ROOT/'scripts/audit_natural_retention_feedback.py').read_text()
    old_start=source.index('        run=np.zeros(768,dtype=np.int64);acquired=')
    old_end=source.index("        action=v('action').copy()",old_start)
    replay='''        run=np.zeros(768,dtype=np.int64);event=np.full(768,-1,dtype=np.int64)
        latch=np.full(768,-1,dtype=np.int64);anchor=np.zeros((768,3),np.float32)
        goal=np.empty_like(base);lift=n('lift_start')[motion];radius=np.abs(n('pd_scale')[3:6])
        for tick in range(202):
            valid=(obj[tick,:,2]-n('initial_height')>=np.float32(.03))&(preclear[tick]>=np.float32(.02))&(tick>=lift)
            run=np.where(valid,run+1,0);trigger=(event<0)&(run>=5);event[trigger]=tick
            active=((arm==2)&(tick>=steps))|((arm==3)&(event>=0));first=active&(latch<0)
            latch[first]=tick;anchor[first]=q[tick,first,3:6]
            target=base[tick].copy();held=np.maximum(np.minimum(anchor,q[tick,:,3:6]+radius),q[tick,:,3:6]-radius)
            target[active,3:6]=held[active];goal[tick]=target
        if not np.array_equal(event,v('feedback_event_tick')) or not np.array_equal(latch,v('rotation_anchor_tick')) or not np.array_equal(anchor,v('rotation_anchor')):raise ValueError('current acquisition and rotation latch replay')
        if not np.allclose(goal,v('target'),atol=1e-5,rtol=0):raise ValueError('rotation-only goal decoding')
        if not np.array_equal(v('target')[...,:3],v('base_target')[...,:3]) or not np.array_equal(v('target')[...,6:],v('base_target')[...,6:]):raise ValueError('translation/finger feedback must remain exact same-state base')
'''
    source=source[:old_start]+replay+source[old_end:]
    changes={
        '[543,544]':'[571,572]',
        'for seed in (543,544):':'for seed in (571,572):',
        "names=('unchanged','early_curl','event_curl','event_wrist_arrest')":"names=('unchanged','duplicate_unchanged','early_rotation_hold','event_rotation_hold')",
        'for s in (543,544)':'for s in (571,572)',
        'for arm in names[2:]':'for arm in names[3:]',
        'for c in names[:2]':'for c in names[:3]',
        'pooled_gain5pp_both':'pooled_gain5pp_all_three',
        'each_seed_no_worse_both':'each_seed_no_worse_all_three',
    }
    for old,new in changes.items():
        if old not in source:raise ValueError('original audit marker drift '+old)
        source=source.replace(old,new)
    source=source.replace('current_state_event_and_native_PD_reconstruction=True','current_state_event_and_native_PD_reconstruction=True,translation_finger_targets_preserved=True,rotation_latches_and_feasible_projection_verified=True')
    marker="        physical=json.loads((directory/'physical_metadata.json').read_text())"
    checked=marker+'''
        import xml.etree.ElementTree as ET
        joints={j.attrib['name']:j for j in ET.parse(ROOT/'third_party/DExplore/dexplore/data/assets/inspire_hand_new/inspire_hand_right.urdf').getroot().findall('joint')}
        names=physical['native_dof_names']
        if len(names)!=18 or names[:6]!=['joint'+str(i) for i in range(1,7)]:raise ValueError('native wrist coordinate order')
        if any(joints[name].attrib['type']!='prismatic' for name in names[:3]) or any(joints[name].attrib['type'] not in ('revolute','continuous') for name in names[3:6]):raise ValueError('URDF translation/rotation responsibility mapping')
'''
    if source.count(marker)!=1:raise ValueError('native metadata audit drift')
    source=source.replace(marker,checked)
    exec(compile(source,str(ROOT/'scripts/audit_natural_retention_feedback.py'),'exec'),{'__name__':'__main__','__file__':str(ROOT/'scripts/audit_natural_retention_feedback.py')})

if __name__=='__main__':main()
