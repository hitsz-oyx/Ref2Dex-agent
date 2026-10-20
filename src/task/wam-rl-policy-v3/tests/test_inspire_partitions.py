from wam_rl_policy_v3.inspire_partitions import partition_inspire


def test_six_ranks_cover_all_training_objects_and_motions_without_dev_or_duplicates():
    records=[dict(object='object-%02d'%obj,sequence='%d-%d'%(obj,motion),split='train') for obj in range(50) for motion in range(3)]
    records += [dict(object='object-00',sequence='dev-only',split='dev')]
    ranks=partition_inspire(records,6,246)
    assert [len(r['object_names']) for r in ranks]==[9,9,8,8,8,8]
    assert ranks==partition_inspire(records,6,246)
    objects=[o for r in ranks for o in r['object_names']]
    sequences=[s for r in ranks for s in r['sequence_names']]
    assert len(objects)==len(set(objects))==50
    assert len(sequences)==len(set(sequences))==150 and 'dev-only' not in sequences
    for rank in ranks:
        assert set(rank['sequence_names'])=={r['sequence'] for r in records if r['split']=='train' and r['object'] in rank['object_names']}
