import os
from typing import Callable

import pytest
from picsellia.types.enums import InferenceType


class TestDatasetCollection:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download(
        self, dataset_type: InferenceType, mock_dataset_collection: Callable
    ):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
        dataset_collection.download()
        for dataset_context in dataset_collection:
            assert dataset_context is not None
            download_path = os.path.join(
                dataset_context.dataset_extraction_path, "images"
            )
            assert os.path.exists(download_path)
            assert len(os.listdir(download_path)) == len(dataset_context.multi_asset)
            filenames = [
                asset.id_with_extension for asset in dataset_context.multi_asset
            ]
            for file in os.listdir(download_path):
                assert file in filenames
            assert dataset_context.coco_file is not None
            for category in dataset_context.coco_file.categories:
                assert category.name in dataset_context.labelmap.keys()
            assert len(dataset_context.coco_file.images) == len(
                dataset_context.multi_asset
            )
