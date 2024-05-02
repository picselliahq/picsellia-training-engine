from typing import Callable

import pytest
from src.models.steps.data_preparation.classification_dataset_context_preparator import (
    ClassificationDatasetContextPreparator,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_classification_dataset_context_preparator(
    mock_dataset_context: Callable,
) -> Callable:
    def _mock_classification_dataset_context_preparator(
        dataset_metadata: DatasetTestMetadata,
    ) -> ClassificationDatasetContextPreparator:
        dataset_context = mock_dataset_context(dataset_metadata=dataset_metadata)
        dataset_context.download_assets()
        return ClassificationDatasetContextPreparator(dataset_context=dataset_context)

    return _mock_classification_dataset_context_preparator
