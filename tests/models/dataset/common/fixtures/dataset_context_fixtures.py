import os
from typing import Callable, Dict, Optional

import pytest
from picsellia import Label
from picsellia.sdk.asset import MultiAsset

from src.models.dataset.common.dataset_context import DatasetContext
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_dataset_context(
    destination_path: str, mock_dataset_version: Callable
) -> Callable:
    """
    Fixture to mock a DatasetContext for testing, simulating assets and labelmap.

    Args:
        destination_path (str): Path where the assets are downloaded.
        mock_dataset_version (Callable): Mock for the dataset version.

    Returns:
        Callable: Function to create a DatasetContext for testing purposes.
    """

    def _mock_dataset_context(
        dataset_metadata: DatasetTestMetadata,
        assets: Optional[MultiAsset] = None,
        labelmap: Optional[Dict[str, Label]] = None,
    ) -> DatasetContext:
        dataset_version = mock_dataset_version(dataset_metadata=dataset_metadata)
        dataset_context = DatasetContext(
            dataset_name=dataset_metadata.attached_name,
            dataset_version=dataset_version,
            assets=assets,
            labelmap=labelmap,
        )
        dataset_context.images_dir = os.path.join(destination_path, "images")
        dataset_context.annotations_dir = os.path.join(destination_path, "annotations")
        return dataset_context

    return _mock_dataset_context
