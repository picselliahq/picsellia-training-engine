from typing import Callable

import pytest

from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.steps.data_validation.common.classification_dataset_context_validator import (
    ClassificationDatasetContextValidator,
)
from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


@pytest.fixture
def mock_dataset_collection_validator() -> Callable:
    """
    Fixture to create a DatasetCollectionValidator for generic dataset collections.

    Returns:
        Callable: A function that takes a dataset collection and returns a DatasetCollectionValidator.
    """

    def _dataset_collection_validator(
        dataset_collection: DatasetCollection,
    ) -> DatasetCollectionValidator:
        return DatasetCollectionValidator(
            dataset_collection=dataset_collection,
            dataset_context_validator=DatasetContextValidator,
        )

    return _dataset_collection_validator


@pytest.fixture
def mock_classification_dataset_collection_validator() -> Callable:
    """
    Fixture to create a DatasetCollectionValidator specifically for classification datasets.

    Returns:
        Callable: A function that takes a classification dataset collection and returns a DatasetCollectionValidator.
    """

    def _classification_dataset_collection_validator(
        classification_dataset_collection: DatasetCollection,
    ) -> DatasetCollectionValidator:
        return DatasetCollectionValidator(
            dataset_collection=classification_dataset_collection,
            dataset_context_validator=ClassificationDatasetContextValidator,
        )

    return _classification_dataset_collection_validator
