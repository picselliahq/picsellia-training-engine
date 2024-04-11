from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.steps.data_validation.utils.classification_dataset_validator import (
    ClassificationDatasetValidator,
)
from src.steps.data_validation.utils.dataset_validator import DatasetValidator


@pytest.fixture
def mock_dataset_validator(mock_dataset_collection: Callable) -> Callable:
    def _dataset_validator(dataset_type: InferenceType) -> DatasetValidator:
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
        dataset_collection.download()
        return DatasetValidator(dataset_collection=dataset_collection)

    return _dataset_validator


@pytest.fixture
def classification_dataset_validator(
    mock_dataset_collection: Callable,
) -> ClassificationDatasetValidator:
    classification_dataset_collection = mock_dataset_collection(
        dataset_type=InferenceType.CLASSIFICATION
    )
    classification_dataset_collection.download()
    return ClassificationDatasetValidator(
        dataset_collection=classification_dataset_collection
    )
