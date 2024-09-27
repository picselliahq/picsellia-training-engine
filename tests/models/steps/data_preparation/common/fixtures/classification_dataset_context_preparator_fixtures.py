import os
from typing import Callable
import tempfile
import pytest
from src.models.steps.data_preparation.common.classification_dataset_context_preparator import (
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
        # Create a temporary directory for downloading assets and organizing files
        download_dir = tempfile.mkdtemp()
        organize_dir = tempfile.mkdtemp()

        # Initialize the dataset context and download assets and COCO file
        dataset_context = mock_dataset_context(dataset_metadata=dataset_metadata)
        dataset_context.download_assets(
            destination_path=os.path.join(download_dir, "images")
        )
        dataset_context.download_and_build_coco_file(
            destination_path=os.path.join(download_dir, "annotations")
        )

        # Return the preparator with the dataset context and destination path
        return ClassificationDatasetContextPreparator(
            dataset_context=dataset_context, destination_path=organize_dir
        )

    return _mock_classification_dataset_context_preparator
