from typing import Callable

import pytest

from src.models.dataset.dataset_context import DatasetContext
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata
from tests.steps.fixtures.initialize_integration_tests_fixtures import (
    get_multi_asset,
    get_labelmap,
)


@pytest.fixture
def mock_dataset_context(
    destination_path: str, mock_dataset_version: Callable
) -> Callable:
    def _mock_dataset_context(dataset_metadata: DatasetTestMetadata) -> DatasetContext:
        dataset_version = mock_dataset_version(dataset_metadata=dataset_metadata)
        dataset_context = DatasetContext(
            dataset_name=dataset_metadata.attached_name,
            dataset_version=dataset_version,
            multi_asset=get_multi_asset(dataset_version=dataset_version),
            labelmap=get_labelmap(dataset_version=dataset_version),
            destination_path=destination_path,
        )
        return dataset_context

    return _mock_dataset_context
