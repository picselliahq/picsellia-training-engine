import os

import pytest
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName


class TestDatasetCollection:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_download(self, dataset_type: InferenceType, mock_dataset_collection):
        dataset_collection = mock_dataset_collection(dataset_type=dataset_type)
        dataset_collection.download()
        for context_name in [
            DatasetSplitName.TRAIN.value,
            DatasetSplitName.VAL.value,
            DatasetSplitName.TEST.value,
        ]:
            context = getattr(dataset_collection, context_name)
            assert context is not None
            download_path = os.path.join(context.dataset_extraction_path, "images")
            assert os.path.exists(download_path)
            assert len(os.listdir(download_path)) == len(context.multi_asset)
            filenames = [asset.id_with_extension for asset in context.multi_asset]
            for file in os.listdir(download_path):
                assert file in filenames
            assert context.coco_file is not None
            for category in context.coco_file.categories:
                assert category.name in context.labelmap.keys()
            assert len(context.coco_file.images) == len(context.multi_asset)
