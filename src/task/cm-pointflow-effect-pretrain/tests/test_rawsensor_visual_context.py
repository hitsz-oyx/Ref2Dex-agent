"""Current-frame temporal isolation and matched information interventions."""
import importlib.util
from pathlib import Path

import numpy as np

script = Path(__file__).resolve().parents[1] / 'tools/run/train_rawsensor_visual_context.py'
spec = importlib.util.spec_from_file_location('visual_screen', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Windows:
    index = [('record', 0), ('record', 4)]

    def __len__(self):
        return len(self.index)


def test_current_rgb_does_not_read_future_frames():
    features = {'record': np.arange(40 * 512, dtype='float32').reshape(40, 512)}
    before = dict(features=np.zeros((2, 5), dtype='float32'))
    module.add_current_features(before, Windows(), features)
    # The original current frames are 3 and 7. Change all other images, including
    # targets and history; per-window current-image features must stay identical.
    features['record'][[i for i in range(40) if i not in (3, 7)]] = -999
    after = dict(features=np.zeros((2, 5), dtype='float32'))
    module.add_current_features(after, Windows(), features)
    np.testing.assert_array_equal(before['features'], after['features'])
    np.testing.assert_array_equal(after['features'][:, 5:], features['record'][[3, 7]])


def test_matched_ablation_keeps_observed_inputs_and_swaps_only_named_block():
    features = np.arange(3 * 12, dtype='float32').reshape(3, 12)
    donors = np.asarray([1, 2, 0])
    inputs = module.variants(features, donors, 5, 9)
    for value in inputs.values():
        np.testing.assert_array_equal(value[:, :5], features[:, :5])
    np.testing.assert_array_equal(inputs['history'][:, 5:], 0)
    np.testing.assert_array_equal(inputs['future_hand'][:, 5:9], features[:, 5:9])
    np.testing.assert_array_equal(inputs['future_hand'][:, 9:], 0)
    np.testing.assert_array_equal(inputs['history_rgb'][:, 5:9], 0)
    np.testing.assert_array_equal(inputs['history_rgb'][:, 9:], features[:, 9:])
    np.testing.assert_array_equal(inputs['future_hand_rgb'], features)
    np.testing.assert_array_equal(inputs['shuffled_future_hand_rgb'][:, 5:9], features[donors, 5:9])
    np.testing.assert_array_equal(inputs['shuffled_future_hand_rgb'][:, 9:], features[:, 9:])
    np.testing.assert_array_equal(inputs['shuffled_rgb'][:, 5:9], features[:, 5:9])
    np.testing.assert_array_equal(inputs['shuffled_rgb'][:, 9:], features[donors, 9:])
