import pytest
import torch
from torchvision import transforms

from src.steps.model_loader.processing.processing_diversified_data_extractor_model_loader import (
    OpenClipEmbeddingModel,
)


class MockCLIP:
    def encode_image(self, _):
        return torch.randn(1, 512)


@pytest.fixture
def mock_open_clip_model_device():
    return "cpu"


@pytest.fixture
def mock_open_clip_model():
    return MockCLIP()


@pytest.fixture
def mock_preprocessing():
    return transforms.Compose([transforms.ToTensor()])


@pytest.fixture
def mock_open_clip_embedding_model(
    mock_open_clip_model, mock_preprocessing, mock_open_clip_model_device
):
    return OpenClipEmbeddingModel(
        model=mock_open_clip_model,  # noqa
        preprocessing=mock_preprocessing,
        device=mock_open_clip_model_device,
    )
