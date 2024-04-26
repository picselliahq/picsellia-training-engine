from typing import Callable

import pytest

from src.models.dataset.dataset_collection import DatasetCollection
from src.models.dataset.dataset_context import DatasetContext
from src.steps.data_validation.utils.classification_dataset_context_validator import (
    ClassificationDatasetContextValidator,
)
from src.steps.data_validation.utils.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.steps.data_validation.utils.dataset_context_validator import (
    DatasetContextValidator,
)


@pytest.fixture
def mock_dataset_context_validator() -> Callable:
    def _dataset_context_validator(
        dataset_context: DatasetContext,
    ) -> DatasetContextValidator:
        return DatasetContextValidator(dataset_context=dataset_context)

    return _dataset_context_validator


@pytest.fixture
def mock_dataset_collection_validator() -> Callable:
    def _dataset_collection_validator(
        dataset_collection: DatasetCollection,
    ) -> DatasetCollectionValidator:
        return DatasetCollectionValidator(
            dataset_collection=dataset_collection,
            dataset_context_validator=DatasetContextValidator,
        )

    return _dataset_collection_validator


@pytest.fixture
def mock_classification_dataset_context_validator() -> Callable:
    def _classification_dataset_context_validator(
        dataset_context: DatasetContext,
    ) -> ClassificationDatasetContextValidator:
        return ClassificationDatasetContextValidator(dataset_context=dataset_context)

    return _classification_dataset_context_validator


@pytest.fixture
def mock_classification_dataset_collection_validator() -> Callable:
    def _classification_dataset_collection_validator(
        classification_dataset_collection: DatasetCollection,
    ) -> DatasetCollectionValidator:
        return DatasetCollectionValidator(
            dataset_collection=classification_dataset_collection,
            dataset_context_validator=ClassificationDatasetContextValidator,
        )

    return _classification_dataset_collection_validator
