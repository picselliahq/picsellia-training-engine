import os
import unittest
from unittest.mock import patch

from utils import extract_file_name, get_metrics_curves, get_weights_and_config


class TestYolov8Segmentation(unittest.TestCase):
    final_run_path = "/home/user/final_run"

    @classmethod
    def setUpClass(cls) -> None:
        pass

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def test_extract_file_name(self):
        file_path = "/home/user/filename.ext"
        result = extract_file_name(file_path)
        expected = "filename"
        self.assertEqual(result, expected)

    def test_extract_file_name_no_extension(self):
        file_path = "/home/user/filename"
        result = extract_file_name(file_path)
        expected = "filename"
        self.assertEqual(result, expected)

    @patch("os.path.isfile")
    def test_get_metrics_curves_all_paths_exist(self, mock_os_is_file):
        mock_os_is_file.return_value = True
        result = get_metrics_curves(self.final_run_path)
        expected = (
            "/home/user/final_run/confusion_matrix.png",
            "/home/user/final_run/F1_curve.png",
            "/home/user/final_run/labels_correlogram.jpg",
            "/home/user/final_run/labels.jpg",
            "/home/user/final_run/P_curve.png",
            "/home/user/final_run/PR_curve.png",
            "/home/user/final_run/R_curve.png",
            "/home/user/final_run/BoxF1_curve.png",
            "/home/user/final_run/BoxP_curve.png",
            "/home/user/final_run/BoxPR_curve.png",
            "/home/user/final_run/BoxR_curve.png",
            "/home/user/final_run/MaskF1_curve.png",
            "/home/user/final_run/MaskP_curve.png",
            "/home/user/final_run/MaskPR_curve.png",
            "/home/user/final_run/MaskR_curve.png",
        )
        self.assertEqual(result, expected)

    @patch("os.path.isfile")
    def test_get_metrics_curves_none_exist(self, mock_os_is_file):
        mock_os_is_file.return_value = False
        result = get_metrics_curves(self.final_run_path)
        expected = (None,) * 15
        self.assertEqual(result, expected)

    def test_best_weights_file_exists(self):
        best_weights_path = os.path.join(self.final_run_path, "weights", "best.pt")
        with unittest.mock.patch("os.path.isfile", return_value=True):
            result = get_weights_and_config(self.final_run_path)
        expected = (best_weights_path, None)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
