"""Decision-critical leakage, visibility and physical-clock contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK / 'src'))
from oakink_wm.video_points import sample_window, collate_video, balanced_flow_loss, flow_metrics, VideoWindows


def sequence():
    t = np.arange(35) * 2 / (60000 / 1001)
    xyz = np.stack((np.linspace(-.2, .2, 40), np.zeros(40), np.ones(40)), -1)
    velocity = np.zeros((40, 3))
    velocity[:20, 0] = .05
    return xyz[None] + t[:, None, None] * velocity, np.ones((35, 40), bool), np.repeat([1, 0], 20), t


def test_future_labels_and_visibility_cannot_change_observed_inputs():
    points, valid, kind, ts = sequence()
    sample = sample_window(points, valid, kind, ts, 0)
    changed, masked = points.copy(), valid.copy()
    changed[4:] = np.nan
    masked[4:] = False
    other = sample_window(changed, masked, kind, ts, 0)
    for key in ('xyz', 'features', 'velocity', 'selected_track_ids'):
        np.testing.assert_array_equal(sample[key], other[key])
    assert not other['target_valid'].any()
    assert np.isfinite(other['target_flow']).all()


def test_cv_baseline_uses_actual_elapsed_time():
    points, valid, kind, ts = sequence()
    batch = collate_video([sample_window(points, valid, kind, ts, 0)])
    cv = batch['velocity'][:, :, None] * batch['elapsed'][:, None, :, None]
    assert abs(batch['elapsed'][0, -1].item() - .8008) < 1e-6
    metrics = flow_metrics(cv, batch)
    assert metrics['object/h24'][0] < 1e-6
    assert metrics['background/all'][0] == 0


def test_masked_nonfinite_labels_are_excluded_before_loss_arithmetic():
    points, valid, kind, ts = sequence()
    batch = collate_video([sample_window(points, valid, kind, ts, 0)])
    pred = torch.zeros_like(batch['target_flow'], requires_grad=True)
    batch['target_valid'][0, 0] = False
    baseline = balanced_flow_loss(pred, batch)
    batch['target_flow'][0, 0] = torch.nan
    actual = balanced_flow_loss(pred, batch)
    torch.testing.assert_close(actual, baseline)
    actual.backward()
    assert torch.isfinite(pred.grad).all()
    assert not pred.grad[0, 0].any()


def test_history_invalid_tracks_are_not_silently_imputed():
    points, valid, kind, ts = sequence()
    valid[1, :5] = False
    with pytest.raises(ValueError, match='history-valid'):
        sample_window(points, valid, kind, ts, 0)


def test_background_density_cannot_change_kind_balanced_loss():
    points, valid, kind, ts = sequence()
    batch = collate_video([sample_window(points, valid, kind, ts, 0)])
    prediction = torch.zeros_like(batch['target_flow'])
    original = balanced_flow_loss(prediction, batch)
    repeated = {key: torch.cat((value, value[:, 20:].repeat_interleave(10, dim=1)), dim=1)
                if value.ndim >= 2 and value.shape[1] == 40 else value for key, value in batch.items()}
    longer = torch.zeros_like(repeated['target_flow'])
    torch.testing.assert_close(balanced_flow_loss(longer, repeated), original)


def test_window_blocks_preserve_birth_specific_identity_and_clock(tmp_path):
    import json
    points, valid, kind, ts = sequence()
    # Two independently born windows, with distinct origins and start times.
    blocks = np.stack((points[:28], points[2:30] + 10))
    clocks = np.stack((ts[:28], ts[2:30]))
    np.savez_compressed(tmp_path / 'scene.npz', points=blocks,
                        valid=np.stack((valid[:28], valid[2:30])),
                        kind=np.stack((kind, kind)), timestamps=clocks)
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        status='QUALIFIED_VIDEO_PROBE_ONLY', protocol=dict(points_per_kind=128),
        sequences=[dict(scene='scene', split='train', file='scene.npz', window_starts=[0, 1])])) )
    dataset = VideoWindows(tmp_path, 'train')
    expected = sample_window(blocks[1], valid[2:30], kind, clocks[1], 0)
    for key in expected:
        np.testing.assert_array_equal(dataset[1][key], expected[key])
    # Change a different birth's future; this birth's observed input is stable.
    dataset.sequences['scene']['points'][0, 4:] = np.nan
    np.testing.assert_array_equal(dataset[1]['features'], expected['features'])
    dataset.manifest['protocol']['supervised_horizon'] = 8
    short = dataset[1]
    np.testing.assert_array_equal(short['features'], expected['features'])
    assert short['target_valid'][:, :8].all()
    assert not short['target_valid'][:, 8:].any()
    batch = collate_video([short])
    pred = torch.zeros_like(batch['target_flow'], requires_grad=True)
    balanced_flow_loss(pred, batch).backward()
    assert not pred.grad[:, :, 8:].any()
    assert flow_metrics(pred.detach(), batch, horizon=8)['object/h8'][1] == 20
