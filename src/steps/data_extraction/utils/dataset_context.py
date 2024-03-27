import os

from picsellia import DatasetVersion
from picsellia.sdk.asset import MultiAsset


class DatasetContext:
    def __init__(
        self,
        dataset_name: str,
        dataset_version: DatasetVersion,
        multi_asset: MultiAsset,
        labelmap: dict,
        destination_path: str,
    ):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.multi_asset = multi_asset
        self.dataset_extraction_path = os.path.join(destination_path, self.dataset_name)
        self.labelmap = labelmap
        self.image_dir = None
        self.coco_file = None

    def download_assets(self) -> None:
        self.image_dir = os.path.join(self.dataset_extraction_path, "images")
        self.multi_asset.download(target_path=self.image_dir, use_id=True)

    def download_coco_file(self) -> None:
        self.coco_file = self.dataset_version.build_coco_file_locally(
            assets=self.multi_asset,
            use_id=True,
        )
