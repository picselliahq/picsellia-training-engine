from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from src.models.dataset.common.dataset_collection import DatasetCollection
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_dataset_collection(mock_dataset_context: Callable) -> Callable:
    """
    Fixture to mock a DatasetCollection for testing purposes.

    This fixture creates a collection of mock dataset contexts for training, validation, and testing splits.

    Args:
        mock_dataset_context (Callable): The fixture that provides a mocked DatasetContext.

    Returns:
        Callable: A function that returns a DatasetCollection when called with a dataset type.
    """

    def _mock_dataset_collection(dataset_type: InferenceType) -> DatasetCollection:
        # Creating mock dataset contexts for each dataset split (train, val, test)
        train_dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN, dataset_type=dataset_type
            )
        )
        val_dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.VAL, dataset_type=dataset_type
            )
        )
        test_dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TEST, dataset_type=dataset_type
            )
        )
        # Returning a DatasetCollection that contains all three splits
        dataset_collection = DatasetCollection(
            [train_dataset_context, val_dataset_context, test_dataset_context]
        )
        return dataset_collection

    return _mock_dataset_collection
