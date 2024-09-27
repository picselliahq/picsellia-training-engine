import os
from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetContext:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_assets(
        self, dataset_type: InferenceType, mock_dataset_context: Callable
    ):
        # Create a mock dataset context with assets
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN, dataset_type=dataset_type
            )
        )

        # Case 1: Download assets to the 'images' subdirectory within destination_path when assets are present
        dataset_context.download_assets(destination_path=dataset_context.images_dir)

        # Check that the 'images' directory exists and contains the downloaded files
        image_dir = dataset_context.images_dir
        assert os.path.exists(image_dir)
        assert len(os.listdir(image_dir)) == len(dataset_context.assets)

        # Verify that each downloaded file matches the expected asset filenames
        filenames = [asset.id_with_extension for asset in dataset_context.assets]
        for file in os.listdir(image_dir):
            assert file in filenames

        # Case 2: When assets are None, ensure the entire dataset is downloaded
        dataset_context.assets = (
            None  # Setting assets to None to trigger full dataset download
        )
        dataset_context.download_assets(destination_path=dataset_context.images_dir)

        # Verify that the directory exists and that all assets were downloaded
        assert os.path.exists(image_dir)
        all_assets = dataset_context.dataset_version.list_assets()
        assert len(os.listdir(image_dir)) == len(all_assets)

        # Verify the filenames again for the full dataset download
        filenames = [asset.id_with_extension for asset in all_assets]
        for file in os.listdir(image_dir):
            assert file in filenames

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_and_build_coco_file(
        self,
        dataset_type: InferenceType,
        mock_dataset_context: Callable,
    ):
        # Create a mock dataset context
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN, dataset_type=dataset_type
            )
        )

        # Download and build the COCO file in the 'annotations' subdirectory
        dataset_context.download_and_build_coco_file(
            destination_path=dataset_context.annotations_dir
        )

        # Assert that the COCO file was created and contains the expected labels
        assert dataset_context.coco_file is not None
        for category in dataset_context.coco_file.categories:
            assert category.name in dataset_context.labelmap.keys()

        # Verify that the number of images in the COCO file matches the dataset assets
        assert len(dataset_context.coco_file.images) == len(dataset_context.assets)

        # Case when assets are None, ensure the COCO file is built with all dataset assets
        dataset_context.assets = None  # Trigger the full listing of assets
        dataset_context.download_and_build_coco_file(
            destination_path=dataset_context.annotations_dir
        )
        assert dataset_context.coco_file is not None
        all_assets = dataset_context.dataset_version.list_assets()
        assert len(dataset_context.coco_file.images) == len(all_assets)
