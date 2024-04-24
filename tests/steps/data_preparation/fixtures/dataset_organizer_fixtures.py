from typing import Callable

import pytest
from src.steps.data_preparation.utils.classification_dataset_organizer import (
    ClassificationDatasetOrganizer,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_classification_dataset_organizer(mock_dataset_context: Callable) -> Callable:
    def _mock_classification_dataset_organizer(
        dataset_metadata: DatasetTestMetadata,
    ) -> ClassificationDatasetOrganizer:
        dataset_context = mock_dataset_context(dataset_metadata=dataset_metadata)
        dataset_context.download_assets()
        dataset_context.download_coco_file()
        return ClassificationDatasetOrganizer(dataset_context=dataset_context)

    return _mock_classification_dataset_organizer