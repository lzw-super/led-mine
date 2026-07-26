import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from scripts.bin_image_process import (BAYER_TO_BGR, apply_ccm_to_bgr,
                                       depack_bayer, gray_world_gains, read_bin,
                                       unpack_mipi_raw10)


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

    def test_read_bin_auto_detects_little_endian_raw16(self):
        expected = np.array([[64, 65, 1022, 1023]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'frame.raw'
            expected.astype('<u2').tofile(path)

            actual, detected_format = read_bin(
                path, height=1, width=4, bin_format='auto', raw8_scale=4.0)

        self.assertEqual(detected_format, 'raw16le')
        np.testing.assert_array_equal(actual, expected)

    def test_read_bin_reads_big_endian_raw16_when_requested(self):
        expected = np.array([[64, 65, 1022, 1023]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'frame.raw'
            expected.astype('>u2').tofile(path)

            actual, detected_format = read_bin(
                path, height=1, width=4, bin_format='raw16be', raw8_scale=4.0)

        self.assertEqual(detected_format, 'raw16be')
        np.testing.assert_array_equal(actual, expected)

    def test_gray_world_gains_equalize_red_and_blue_to_green(self):
        packed = np.array([[[0.25]], [[0.5]], [[0.125]], [[0.5]]], dtype=np.float32)

        actual = gray_world_gains(packed)

        np.testing.assert_allclose(actual, [2.0, 1.0, 4.0, 1.0])

    def test_cfa_to_bgr_mapping_preserves_rgb_channel_meaning(self):
        for bayer in ('RGGB', 'BGGR', 'GRBG', 'GBRG'):
            packed = np.empty((4, 8, 8), dtype=np.uint16)
            packed[0].fill(1000)  # R
            packed[1].fill(500)   # G1
            packed[2].fill(250)   # B
            packed[3].fill(500)   # G2
            mosaic = depack_bayer(packed, bayer)

            bgr = cv2.cvtColor(mosaic, BAYER_TO_BGR[bayer])

            np.testing.assert_array_equal(bgr[3, 3], [250, 500, 1000])

    def test_apply_ccm_operates_in_rgb_order(self):
        bgr = np.array([[[0.1, 0.2, 0.3]]], dtype=np.float32)
        ccm = np.diag([2.0, 1.0, 0.5]).astype(np.float32)

        actual = apply_ccm_to_bgr(bgr, ccm)

        np.testing.assert_allclose(actual, [[[0.05, 0.2, 0.6]]])
