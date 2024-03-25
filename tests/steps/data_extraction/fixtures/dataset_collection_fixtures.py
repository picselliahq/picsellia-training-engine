import pytest

from src.steps.data_extraction.utils.dataset_collection import DatasetCollection


@pytest.fixture
def mock_dataset_collection(
    mock_train_dataset_context, mock_val_dataset_context, mock_test_dataset_context
):
    return DatasetCollection(
        train_context=mock_train_dataset_context,
        val_context=mock_val_dataset_context,
        test_context=mock_test_dataset_context,
    )
