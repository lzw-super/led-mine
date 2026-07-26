import tempfile
import unittest
from pathlib import Path

from scripts.image_process import find_camera_raw_paths, parse_extensions


class ImageProcessPathsTest(unittest.TestCase):
    def test_directory_selection_excludes_headerless_raw(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            directory = Path(tmp_dir)
            (directory / 'camera.DNG').touch()
            (directory / 'camera.ARW').touch()
            (directory / 'frame.raw').touch()
            (directory / 'notes.txt').touch()

            paths = find_camera_raw_paths(directory, parse_extensions('arw,dng'))

        self.assertEqual([path.name for path in paths], ['camera.ARW', 'camera.DNG'])

    def test_headerless_raw_file_has_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'frame.raw'
            path.touch()

            with self.assertRaisesRegex(ValueError, 'bin_image_process'):
                find_camera_raw_paths(path, parse_extensions('arw,dng'))
