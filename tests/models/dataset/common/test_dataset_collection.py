from typing import Callable
from unittest.mock import patch

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetCollection:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)

        # Mocking both methods: download_assets and download_and_build_coco_file
        with patch(
            "src.models.dataset.common.dataset_context.DatasetContext.download_assets"
        ) as mocked_download_assets, patch(
            "src.models.dataset.common.dataset_context.DatasetContext.download_and_build_coco_file"
        ) as mocked_download_coco:
            # Define the destination path
            destination_path = "/mock/destination"

            # Call the download_all method
            dataset_collection.download_all(destination_path=destination_path)

            # Verify that download_assets was called for all dataset splits
            assert mocked_download_assets.call_count == 3
            assert mocked_download_coco.call_count == 3

            # Verify that the assets were downloaded to the correct subdirectory for each split
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/train/images",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/val/images",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/test/images",
                use_id=None,
                skip_asset_listing=False,
            )

            # Verify that the annotations were downloaded to the correct subdirectory for each split
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/train/annotations",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/val/annotations",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/test/annotations",
                use_id=None,
                skip_asset_listing=False,
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_with_empty_assets(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)

        # Setting the assets to None for one of the dataset contexts
        dataset_collection["train"].assets = None

        # Mocking both methods: download_assets and download_and_build_coco_file
        with patch(
            "src.models.dataset.common.dataset_context.DatasetContext.download_assets"
        ) as mocked_download_assets, patch(
            "src.models.dataset.common.dataset_context.DatasetContext.download_and_build_coco_file"
        ) as mocked_download_coco:
            # Define the destination path
            destination_path = "/mock/destination"

            # Call the download_all method
            dataset_collection.download_all(destination_path=destination_path)

            # Verify that download_assets was called for all dataset splits
            assert mocked_download_assets.call_count == 3
            assert mocked_download_coco.call_count == 3

            # Verify that the assets were downloaded to the correct subdirectory for each split
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/train/images",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/val/images",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_assets.assert_any_call(
                destination_path="/mock/destination/test/images",
                use_id=None,
                skip_asset_listing=False,
            )

            # Verify that the annotations were downloaded to the correct subdirectory for each split
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/train/annotations",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/val/annotations",
                use_id=None,
                skip_asset_listing=False,
            )
            mocked_download_coco.assert_any_call(
                destination_path="/mock/destination/test/annotations",
                use_id=None,
                skip_asset_listing=False,
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_getitem(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)

        # Check that the correct dataset context is returned for each split
        assert dataset_collection["train"] is dataset_collection.datasets["train"]
        assert dataset_collection["val"] is dataset_collection.datasets["val"]
        assert dataset_collection["test"] is dataset_collection.datasets["test"]

        # Test that a KeyError is raised for a nonexistent dataset split
        with pytest.raises(KeyError):
            _ = dataset_collection["nonexistent"]

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_setitem(
        self,
        dataset_type: InferenceType,
        mock_dataset_context: Callable,
        mock_dataset_collection: Callable,
    ):
        # Create new mock dataset contexts for train and val
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

        # Set new dataset contexts for train and val splits
        dataset_collection["train"] = new_train_dataset_context
        dataset_collection["val"] = new_val_dataset_context

        # Check that the new dataset contexts were correctly set
        assert dataset_collection["train"] is new_train_dataset_context
        assert dataset_collection["val"] is new_val_dataset_context
