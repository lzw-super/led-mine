import unittest

from scripts.evaluate_gpu_batched_crop_merge_sid import summarize


class GpuBatchedEvaluationSummaryTest(unittest.TestCase):
    def test_summary_reports_dataset_total_and_end_to_end_times(self):
        rows = []
        for ratio, tiled_time, load_time in ((100, 0.2, 0.1), (250, 0.3, 0.15)):
            rows.append({
                'ratio': ratio,
                'method': 'crop',
                'tile_batch_size': 18,
                'effective_tile_batch_size': 18,
                'cropped_height': 1152,
                'cropped_width': 2112,
                'crop_retained_percent': 80.29,
                'tile_count': 18,
                'network_launch_count': 1,
                'processed_pixel_ratio': 1.939,
                'input_psnr_db': 30.0,
                'full_psnr_db': 40.0,
                'tiled_psnr_db': 40.0,
                'psnr_vs_full_db': float('inf'),
                'mae_vs_full': 0.0,
                'max_abs_vs_full': 0.0,
                'estimated_gmacs': 986.9,
                'full_time_s': 0.1,
                'tiled_time_s': tiled_time,
                'data_load_time_s': load_time,
                'time_ratio_vs_full': tiled_time / 0.1,
                'speedup_vs_batch1': None,
                'throughput_tflops_s': 9.0,
                'full_peak_memory_mb': 3388.9,
                'tiled_peak_memory_mb': 6529.6,
            })

        overall = next(
            row for row in summarize(rows)
            if row['scope'] == 'all')

        self.assertAlmostEqual(overall['tiled_time_s'], 0.25)
        self.assertAlmostEqual(overall['total_tiled_time_s'], 0.5)
        self.assertAlmostEqual(overall['total_data_load_time_s'], 0.25)
        self.assertAlmostEqual(overall['total_tiled_end_to_end_time_s'], 0.75)
        self.assertAlmostEqual(overall['total_full_time_s'], 0.2)
        self.assertAlmostEqual(overall['total_full_end_to_end_time_s'], 0.45)


if __name__ == '__main__':
    unittest.main()
