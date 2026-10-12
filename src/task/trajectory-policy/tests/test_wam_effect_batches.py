from types import SimpleNamespace

import numpy as np

from trajectory_policy.wam_effect_data import EffectBatches


def corpus():
    source=SimpleNamespace(groups=[np.arange(100),np.arange(100,140),np.array([],dtype=int)])
    return SimpleNamespace(source_weights=np.array([5/9,2/9,2/9]),sources=[source]*3,
                           offsets=np.array([0,140,280,420]))


def test_mixture_and_admitted_robot_roles_are_fixed_on_every_rank():
    for rank in range(4):
        batches=list(EffectBatches(corpus(),{'recovery0':8,'recovery1':16,'recovery2':32},rank,1,2))
        for keys in batches:
            assert len(keys)==100
            assert sum(k[0]=='main' for k in keys)==60
            nominal=[k for k in keys if k[0]=='nominal']
            assert len(nominal)==25 and all(0<=k[1]<=518 and 0<=k[2]<16 for k in nominal)
            for name,takeover in [('recovery0',8),('recovery1',16),('recovery2',32)]:
                selected=[k for k in keys if k[0]==name]
                assert len(selected)==5 and all(takeover<=k[1]<=104 and k[2]==3 for k in selected)


def test_resume_uses_same_occurrences_and_ranks_have_distinct_streams():
    takeovers={'recovery0':8,'recovery1':16,'recovery2':32}
    full=list(EffectBatches(corpus(),takeovers,0,1,4))
    resumed=list(EffectBatches(corpus(),takeovers,0,3,4))
    assert resumed==full[2:]
    other=list(EffectBatches(corpus(),takeovers,1,1,4))
    assert other!=full
