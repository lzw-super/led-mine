from types import SimpleNamespace
import unittest

import numpy as np

from led.data.raw_utils import get_white_level_per_channel


class RawWhiteLevelTest(unittest.TestCase):
    def test_white_level_falls_back_to_scalar_for_dng_without_channel_metadata(self):
        raw = SimpleNamespace(camera_white_level_per_channel=None, white_level=1023)

        actual = get_white_level_per_channel(raw)

        np.testing.assert_array_equal(actual, np.full(4, 1023, dtype=np.float32))

    def test_white_level_keeps_camera_specific_channel_values(self):
        raw = SimpleNamespace(
            camera_white_level_per_channel=[15360, 15361, 15362, 15363],
            white_level=16383,
        )

        actual = get_white_level_per_channel(raw)

        np.testing.assert_array_equal(
            actual, np.array([15360, 15361, 15362, 15363], dtype=np.float32))
