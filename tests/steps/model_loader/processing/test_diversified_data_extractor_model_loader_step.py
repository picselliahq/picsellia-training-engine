import numpy as np
import torch
from PIL import Image

from src.steps.model_loading.processing.processing_diversified_data_extractor_model_loader import (
    SupportedEmbeddingModels,
    is_embedding_model_name_valid,
)


class TestDiversifiedDataExtractorModelLoaderStep:
    def test_is_embedding_model_name_valid_match(
        self,
    ):
        source = "openclip"
        target = SupportedEmbeddingModels.OPENCLIP
        assert is_embedding_model_name_valid(
            source, target
        ), "The function should return True for matching model names."

    def test_is_embedding_model_name_valid_no_match(
        self,
    ):
        source = "resnet"
        target = SupportedEmbeddingModels.OPENCLIP
        assert not is_embedding_model_name_valid(
            source, target
        ), "The function should return False for non-matching model names."

    def test_is_embedding_model_name_valid_case_insensitivity(self):
        source = "OpenCLIP"
        target = SupportedEmbeddingModels.OPENCLIP
        assert is_embedding_model_name_valid(
            source, target
        ), "The function should return True, being case-insensitive."

    def test_apply_preprocessing(
        self, mock_open_clip_embedding_model, mock_open_clip_model_device
    ):
        dummy_image = Image.new("RGB", (100, 100), color="red")
        tensor = mock_open_clip_embedding_model.apply_preprocessing(dummy_image)

        assert tensor.device == torch.device(
            mock_open_clip_model_device
        ), "The tensor should be on the specified device"

        assert tensor.ndim == 4, "The tensor should be 4D with unsqueeze"
        assert (
            tensor.size(0) == 1
        ), "The first dimension size should be 1 after unsqueeze"

    def test_encode_image(self, mock_open_clip_embedding_model):
        dummy_image = Image.new("RGB", (100, 100), color="red")
        preprocessed_image = mock_open_clip_embedding_model.apply_preprocessing(
            dummy_image
        )

        encoded = mock_open_clip_embedding_model.encode_image(preprocessed_image)

        assert isinstance(encoded, np.ndarray), "The output should be a NumPy array"
        assert encoded.shape == (
            1,
            512,
        ), "The shape of the encoded array should match the model's output"
