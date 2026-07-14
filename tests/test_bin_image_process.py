import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.bin_image_process import read_bin, unpack_mipi_raw10


def pack_mipi_raw10(pixels):
    """Encode four 10-bit pixels using the same MIPI RAW10 layout we decode."""
    pixels = np.asarray(pixels, dtype=np.uint16)
    return np.array([
        pixels[0] >> 2,
        pixels[1] >> 2,
        pixels[2] >> 2,
        pixels[3] >> 2,
        ((pixels[0] & 0x03) |
         ((pixels[1] & 0x03) << 2) |
         ((pixels[2] & 0x03) << 4) |
         ((pixels[3] & 0x03) << 6)),
    ], dtype=np.uint8)


class BinImageProcessTest(unittest.TestCase):
    def test_unpack_mipi_raw10_restores_all_ten_bits(self):
        expected = np.array([[0, 1, 1022, 1023]], dtype=np.uint16)

        actual = unpack_mipi_raw10(pack_mipi_raw10(expected.ravel()), height=1, width=4)

        np.testing.assert_array_equal(actual, expected)

    def test_read_bin_auto_detects_raw8_and_scales_to_ten_bit_domain(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'frame.bin'
            np.array([1, 2, 3, 4], dtype=np.uint8).tofile(path)

            actual, detected_format = read_bin(
                path, height=1, width=4, bin_format='auto', raw8_scale=4.0)

        self.assertEqual(detected_format, 'raw8')
        np.testing.assert_array_equal(actual, np.array([[4, 8, 12, 16]], dtype=np.float32))

    def test_read_bin_auto_detects_mipi_raw10(self):
        expected = np.array([[100, 200, 300, 400]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'frame.bin'
            pack_mipi_raw10(expected.ravel()).tofile(path)

            actual, detected_format = read_bin(
                path, height=1, width=4, bin_format='auto', raw8_scale=4.0)

        self.assertEqual(detected_format, 'mipi_raw10')
        np.testing.assert_array_equal(actual, expected)
