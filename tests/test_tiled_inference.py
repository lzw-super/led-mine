import unittest

import torch
from torch import nn

from led.utils.tiled_inference import (
    analyze_batched_tiling,
    analyze_tiling,
    batched_tiled_inference,
    build_batch_tile_crop,
    build_overlap_crop_tiles,
    crop_to_batchable_tiles,
    tiled_inference,
)


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

    def test_full_sid_frame_has_expected_crop_workload(self):
        stats = analyze_tiling(
            1424, 2128, tile_size=512, overlap=192, blend='crop', alignment=16)

        self.assertEqual(stats.tile_count, 35)
        self.assertEqual(stats.output_pixels, 1424 * 2128)
        self.assertEqual(stats.processed_pixels, 7_189_760)
        self.assertAlmostEqual(stats.processed_pixel_ratio, 2.3726450959)
        self.assertIn((512, 512, 15), stats.patch_shapes)

    def test_sid_batch_crop_produces_eighteen_equal_tiles(self):
        crop = build_batch_tile_crop(
            1424, 2128, tile_size=512, overlap=192, alignment=16)
        stats = analyze_batched_tiling(
            crop.cropped_height,
            crop.cropped_width,
            tile_size=512,
            overlap=192,
            alignment=16)

        self.assertEqual(crop.crop_box, (128, 1280, 0, 2112))
        self.assertEqual((crop.cropped_height, crop.cropped_width), (1152, 2112))
        self.assertEqual((crop.tile_rows, crop.tile_columns), (3, 6))
        self.assertEqual(stats.tile_count, 18)
        self.assertEqual(stats.patch_shapes, ((512, 512, 18),))
        self.assertEqual(stats.processed_pixels, 18 * 512 * 512)
        self.assertAlmostEqual(stats.processed_pixel_ratio, 1.9393939394)

    def test_batched_modes_reconstruct_identity_after_aligned_crop(self):
        image = torch.rand(2, 4, 70, 91)
        cropped, crop = crop_to_batchable_tiles(
            image, tile_size=32, overlap=16, alignment=8)
        self.assertEqual(crop.crop_box, (0, 64, 0, 80))

        for blend in ('crop', 'hann', 'uniform'):
            for tile_batch_size in (1, 0):
                with self.subTest(blend=blend, tile_batch_size=tile_batch_size):
                    actual = batched_tiled_inference(
                        nn.Identity(),
                        cropped,
                        tile_size=32,
                        overlap=16,
                        blend=blend,
                        alignment=8,
                        tile_batch_size=tile_batch_size)
                    torch.testing.assert_close(actual, cropped, rtol=1e-6, atol=1e-7)

    def test_batched_crop_matches_full_when_halo_covers_receptive_field(self):
        model = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 4, 3, padding=1),
        ).eval()
        image, _ = crop_to_batchable_tiles(
            torch.rand(1, 4, 61, 77),
            tile_size=32,
            overlap=12,
            alignment=1)

        with torch.inference_mode():
            expected = model(image)
        actual = batched_tiled_inference(
            model,
            image,
            tile_size=32,
            overlap=12,
            blend='crop',
            alignment=1,
            tile_batch_size=0)

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_zero_tile_batch_size_sends_all_tiles_in_one_network_batch(self):
        class RecordingIdentity(nn.Module):
            def __init__(self):
                super().__init__()
                self.seen_batch_sizes = []

            def forward(self, tensor):
                self.seen_batch_sizes.append(tensor.shape[0])
                return tensor

        model = RecordingIdentity()
        image = torch.rand(2, 4, 64, 80)
        batched_tiled_inference(
            model,
            image,
            tile_size=32,
            overlap=16,
            blend='crop',
            alignment=8,
            tile_batch_size=0)

        # 3x4 spatial tiles times the two images already in the input batch.
        self.assertEqual(model.seen_batch_sizes, [24])

    def test_batched_inference_rejects_non_grid_shape(self):
        with self.assertRaisesRegex(ValueError, 'crop_to_batchable_tiles'):
            batched_tiled_inference(
                nn.Identity(),
                torch.rand(1, 4, 61, 77),
                tile_size=32,
                overlap=16,
                alignment=1)


if __name__ == '__main__':
    unittest.main()
