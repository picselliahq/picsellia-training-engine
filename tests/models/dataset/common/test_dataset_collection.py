from typing import Callable
from unittest.mock import patch

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from tests.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetCollection:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
        with patch(
            "src.models.dataset.common.dataset_context.DatasetContext.download_assets"
        ) as mocked_download_assets:
            dataset_collection.download_assets()

            assert mocked_download_assets.call_count == 3

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_getitem(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
        assert dataset_collection["train"] is dataset_collection.train
        assert dataset_collection["val"] is dataset_collection.val
        assert dataset_collection["test"] is dataset_collection.test

        with pytest.raises(AttributeError):
            _ = dataset_collection["nonexistent"]

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_setitem(
        self,
        dataset_type: InferenceType,
        mock_dataset_context: Callable,
        mock_dataset_collection,
    ):
        new_train_dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
                attached_name="new_train",
            )
        )
        new_val_dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.VAL,
                dataset_type=dataset_type,
                attached_name="new_val",
            )
        )

        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)

        dataset_collection["train"] = new_train_dataset_context
        dataset_collection["val"] = new_val_dataset_context

        assert dataset_collection["train"] is new_train_dataset_context
        assert dataset_collection["val"] is new_val_dataset_context
