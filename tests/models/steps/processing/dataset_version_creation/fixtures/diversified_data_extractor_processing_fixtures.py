import open_clip
import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from src.models.steps.processing.dataset_version_creation.diversified_data_extractor_processing import (
    DiversifiedDataExtractorProcessing,
)
from src.steps.model_loading.processing.processing_diversified_data_extractor_model_loader import (
    OpenClipEmbeddingModel,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture(scope="session")
def mock_pretrained_weights():
    return "laion400m_e32"


@pytest.fixture(scope="session")
def mock_model_architecture():
    return "ViT-B-16-plus-240"


@pytest.fixture(scope="session")
def mock_open_clip_embedding_model(mock_model_architecture, mock_pretrained_weights):
    (
        model,
        _,
        preprocessing_transformations,
    ) = open_clip.create_model_and_transforms(
        model_name=mock_model_architecture,
        pretrained=mock_pretrained_weights,
    )
    return OpenClipEmbeddingModel(
        model=model, preprocessing=preprocessing_transformations, device="cpu"
    )


@pytest.fixture
def diversified_data_extractor_processing(
    picsellia_client,
    picsellia_default_datalake,
    mock_dataset_context,
    mock_dataset_version,
    mock_open_clip_embedding_model,
):
    return DiversifiedDataExtractorProcessing(
        client=picsellia_client,
        datalake=picsellia_default_datalake,
        input_dataset_context=mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
                attached_name="input_diversified_data_extractor",
            )
        ),
        output_dataset_version=mock_dataset_version(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
                attached_name="output_diversified_data_extractor",
            )
        ),
        embedding_model=mock_open_clip_embedding_model,
        distance_threshold=5.0,
    )
