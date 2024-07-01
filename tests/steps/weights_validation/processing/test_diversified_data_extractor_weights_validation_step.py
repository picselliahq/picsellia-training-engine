from unittest.mock import patch

import pytest

from src.steps.weights_validation.processing.processing_diversified_data_extractor_weights_validator import (
    validate_pretrained_weights,
    validate_model_architecture,
)


class TestDiversifiedDataExtractorWeightsValidationStep:
    def test_validate_pretrained_weights_valid(self):
        """Test that no exception is raised for valid pretrained weights."""
        validate_pretrained_weights("resnet50", "resnet50,vgg16")

    def test_validate_pretrained_weights_invalid(self):
        """Test that a ValueError is raised for invalid pretrained weights."""
        with pytest.raises(ValueError):
            validate_pretrained_weights("alexnet", "resnet50,vgg16")

    def test_validate_model_architecture_valid(self):
        """Test that no exception is raised for valid model architecture."""
        validate_model_architecture("resnet50", "resnet50,vgg16")

    def test_validate_model_architecture_invalid(self):
        """Test that a ValueError is raised for invalid model architecture."""
        with pytest.raises(ValueError):
            validate_model_architecture("alexnet", "resnet50,vgg16")

    def test_validate_model_architecture_huggingface(self):
        """Test that a NotImplementedError is raised for HuggingFace model architecture."""
        with patch("open_clip.factory.HF_HUB_PREFIX", new="hf_"):
            with pytest.raises(NotImplementedError):
                validate_model_architecture("hf_resnet50", "resnet50,vgg16,hf_resnet50")
