import pytest

from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from src.steps.data_extraction.utils.dataset_collection import DatasetCollection


@pytest.fixture
def mock_dataset_collection(mock_dataset_context):
    def _mock_dataset_collection(dataset_type: InferenceType):
        train_dataset_context = mock_dataset_context(
            dataset_split_name=DatasetSplitName.TRAIN.value, dataset_type=dataset_type
        )
        val_dataset_context = mock_dataset_context(
            dataset_split_name=DatasetSplitName.VAL.value, dataset_type=dataset_type
        )
        test_dataset_context = mock_dataset_context(
            dataset_split_name=DatasetSplitName.TEST.value, dataset_type=dataset_type
        )
        dataset_collection = DatasetCollection(
            train_context=train_dataset_context,
            val_context=val_dataset_context,
            test_context=test_dataset_context,
        )
        return dataset_collection

    return _mock_dataset_collection
