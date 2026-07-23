import unittest

import torch
from torch import nn

from led.utils.tiled_inference import build_overlap_crop_tiles, tiled_inference


class TiledInferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2022)

    def test_overlap_crop_matches_full_inference_when_halo_covers_receptive_field(self):
        model = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, 3, padding=1),
        ).eval()
        image = torch.rand(1, 4, 61, 77)

        with torch.inference_mode():
            expected = model(image)
        actual = tiled_inference(
            model, image, tile_size=32, overlap=12, blend='crop', alignment=1)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_weighted_modes_reconstruct_identity_on_non_aligned_shape(self):
        model = nn.Identity()
        image = torch.rand(2, 4, 61, 77)

        for blend in ('hann', 'uniform'):
            with self.subTest(blend=blend):
                actual = tiled_inference(
                    model, image, tile_size=32, overlap=16, blend=blend, alignment=16)
                torch.testing.assert_close(actual, image, rtol=1e-6, atol=1e-7)

    def test_overlap_crop_tiles_cover_every_output_pixel_once(self):
        height, width = 80, 112
        tiles = build_overlap_crop_tiles(height, width, tile_size=64, overlap=32)
        coverage = torch.zeros(height, width, dtype=torch.int32)
        for tile in tiles:
            top, bottom, left, right = tile.output_box
            coverage[top:bottom, left:right] += 1
            input_top, input_bottom, input_left, input_right = tile.input_box
            self.assertLessEqual(input_bottom - input_top, 64)
            self.assertLessEqual(input_right - input_left, 64)

        self.assertTrue(torch.equal(coverage, torch.ones_like(coverage)))

    def test_invalid_alignment_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiples of alignment'):
            tiled_inference(
                nn.Identity(), torch.rand(1, 4, 64, 64),
                tile_size=60, overlap=16, alignment=16)


if __name__ == '__main__':
    unittest.main()
