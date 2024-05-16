import os
from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from tests.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetContext:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_assets(
        self, dataset_type: InferenceType, mock_dataset_context: Callable
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN, dataset_type=dataset_type
            )
        )

        dataset_context.download_assets()
        image_dir = os.path.join(
            dataset_context.destination_path, dataset_context.dataset_name, "images"
        )
        assert os.path.exists(image_dir)
        assert len(os.listdir(image_dir)) == len(dataset_context.multi_asset)
        filenames = [asset.id_with_extension for asset in dataset_context.multi_asset]
        for file in os.listdir(image_dir):
            assert file in filenames

        # empty multi_asset
        dataset_context.multi_asset = None
        with pytest.raises(ValueError):
            dataset_context.download_assets()

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_build_coco_file(
        self,
        dataset_type: InferenceType,
        mock_dataset_context: Callable,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN, dataset_type=dataset_type
            )
        )
        dataset_context.build_coco_file()
        assert dataset_context.coco_file is not None
        for category in dataset_context.coco_file.categories:
            assert category.name in dataset_context.labelmap.keys()
        assert len(dataset_context.coco_file.images) == len(dataset_context.multi_asset)

        # empty multi_asset
        dataset_context.multi_asset = None
        dataset_context.build_coco_file()
        assert dataset_context.coco_file is not None
        assert len(dataset_context.coco_file.images) == len(
            dataset_context.dataset_version.list_assets()
        )
