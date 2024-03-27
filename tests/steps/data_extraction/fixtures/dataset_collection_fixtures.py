from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from tests.steps.data_extraction.utils.conftest import DatasetTestMetadata


@pytest.fixture
def mock_dataset_collection(mock_dataset_context: Callable) -> Callable:
    def _mock_dataset_collection(dataset_type: InferenceType):
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
        dataset_collection = DatasetCollection(
            train_dataset_context=train_dataset_context,
            val_dataset_context=val_dataset_context,
            test_dataset_context=test_dataset_context,
        )
        return dataset_collection

    return _mock_dataset_collection
