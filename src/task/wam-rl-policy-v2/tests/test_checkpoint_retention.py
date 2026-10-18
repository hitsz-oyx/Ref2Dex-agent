from wam_rl_policy_v2.checkpoint_retention import prune_checkpoints


def test_retention_protects_milestones_final_and_foreign_files(tmp_path):
    for n in [1, 22, 500, 522, 542, 562, 582, 600]:
        (tmp_path / ('official-iteration-%d.pt' % n)).write_bytes(b'checkpoint')
    for name in ['best.pt', 'official-iteration-bad.pt', 'official-iteration-620.pt.partial']:
        (tmp_path / name).write_bytes(b'untouched')
    removed = prune_checkpoints(tmp_path, final_iteration=522)
    assert set(removed) == {'official-iteration-1.pt', 'official-iteration-22.pt', 'official-iteration-542.pt'}
    assert (tmp_path / 'official-iteration-500.pt').exists()
    assert (tmp_path / 'official-iteration-522.pt').exists()
    assert (tmp_path / 'best.pt').exists()
    assert (tmp_path / 'official-iteration-620.pt.partial').exists()
    assert prune_checkpoints(tmp_path, final_iteration=522) == []
