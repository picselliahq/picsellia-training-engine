import os

from src.models.dataset.dataset_type import DatasetType


class TestDatasetCollection:
    def test_download(self, mock_dataset_collection):
        mock_dataset_collection.download()
        for context_name in [
            DatasetType.TRAIN.value,
            DatasetType.VAL.value,
            DatasetType.TEST.value,
        ]:
            context = getattr(mock_dataset_collection, context_name)
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
