import os

import pytest
from picsellia.types.enums import InferenceType


class TestDatasetContext:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_assets(self, dataset_type: InferenceType, mock_dataset_context):
        dataset_context = mock_dataset_context(
            dataset_split_name="train", dataset_type=dataset_type
        )

        dataset_context.download_assets()
        download_path = os.path.join(dataset_context.dataset_extraction_path, "images")
        assert os.path.exists(download_path)
        assert len(os.listdir(download_path)) == len(dataset_context.multi_asset)
        filenames = [asset.id_with_extension for asset in dataset_context.multi_asset]
        for file in os.listdir(download_path):
            assert file in filenames

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download_coco_file(
        self,
        dataset_type: InferenceType,
        mock_dataset_context,
    ):
        dataset_context = mock_dataset_context(
            dataset_split_name="train", dataset_type=dataset_type
        )
        dataset_context.download_coco_file()
        assert dataset_context.coco_file is not None
        for category in dataset_context.coco_file.categories:
            assert category.name in dataset_context.labelmap.keys()
        assert len(dataset_context.coco_file.images) == len(dataset_context.multi_asset)
