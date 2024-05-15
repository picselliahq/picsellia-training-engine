import io

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

import requests


class TestDiversifiedDataExtractorProcessing:
    def test_compute_image_tensor_with_openclip(
        self, diversified_data_extractor_processing
    ):
        dummy_image = Image.new("RGB", (100, 100), color="red")
        tensor = diversified_data_extractor_processing.compute_image_tensor(dummy_image)
        assert tensor.shape == (1, 640), "Tensor shape should be (1, 640)"

    @pytest.mark.parametrize("image_mode", ["RGBA", "L", "RGB", "P"])
    def test_fetch_and_prepare_image(
        self, image_mode, diversified_data_extractor_processing
    ):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()

            image_bytes = io.BytesIO()
            Image.new(image_mode, (100, 100)).save(image_bytes, format="PNG")
            image_bytes.seek(0)
            mock_response.raw = image_bytes
            mock_get.return_value.__enter__.return_value = mock_response

            # Call the method to fetch and prepare the image
            image_url = "https://example.com/image.png"
            image = diversified_data_extractor_processing.fetch_and_prepare_image(
                image_url
            )
            assert image is not None, "Image should be fetched and prepared"
            assert image.mode == "RGB", f"Image mode should be RGB, got {image.mode}"

    def test_fetch_and_prepare_image_failure(
        self, diversified_data_extractor_processing
    ):
        with patch(
            "requests.get", side_effect=requests.RequestException("Network error")
        ):
            image_url = "https://example.com/image.png"
            image = diversified_data_extractor_processing.fetch_and_prepare_image(
                image_url
            )

            assert image is None

    def test_get_tqdm_description_string(self, diversified_data_extractor_processing):
        description = diversified_data_extractor_processing.get_tqdm_description_string(
            20, 10, 100
        )
        assert (
            "Batch 3/" in description
        ), "Progress description should indicate the correct batch number"
        assert (
            "(size: 10)" in description
        ), "Progress description should indicate the correct batch size"

    def test_get_tqdm_postfix_string(self, diversified_data_extractor_processing):
        diversified_data_extractor_processing.uploaded_asset_number = 5
        diversified_data_extractor_processing.skipped_error_asset_number = 2
        diversified_data_extractor_processing.skipped_similar_asset_number = 3
        postfix = diversified_data_extractor_processing.get_tqdm_postfix_string()
        assert (
            "Added" in postfix and "Skipped" in postfix
        ), "Postfix should reflect added and skipped assets"
