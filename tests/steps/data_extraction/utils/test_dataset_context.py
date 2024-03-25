import os


class TestDatasetContext:
    def test_download_assets(self, mock_train_dataset_context):
        mock_train_dataset_context.download_assets()
        download_path = os.path.join(
            mock_train_dataset_context.dataset_extraction_path, "images"
        )
        assert os.path.exists(download_path)
        assert len(os.listdir(download_path)) == len(
            mock_train_dataset_context.multi_asset
        )
        filenames = [
            asset.id_with_extension for asset in mock_train_dataset_context.multi_asset
        ]
        for file in os.listdir(download_path):
            assert file in filenames

    def test_download_coco_file(self, mock_train_dataset_context):
        mock_train_dataset_context.download_coco_file()
        assert mock_train_dataset_context.coco_file is not None
        for category in mock_train_dataset_context.coco_file.categories:
            assert category.name in mock_train_dataset_context.labelmap.keys()
        assert len(mock_train_dataset_context.coco_file.images) == len(
            mock_train_dataset_context.multi_asset
        )
