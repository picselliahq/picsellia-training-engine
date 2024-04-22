import os
from typing import Optional

from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset

from src.steps.data_extraction.utils.image_utils import get_labelmap

from picsellia.exceptions import NoDataError


class DatasetContext:
    """
    This class is used to store the context of a dataset, which includes its metadata, paths to its assets and annotations.

    Attributes:
        - dataset_name: The name of the dataset (not on Picsellia).
        - dataset_version: The version of the dataset, as managed by Picsellia.
        - multi_asset: A collection of assets associated with the dataset.
        - labelmap: A mapping from label names to (Picsellia) label objects.
        - destination_path: The root path where the dataset should be stored locally.
        - dataset_extraction_path: The path where the dataset is extracted, including the dataset name.
        - image_dir: The directory where the dataset images are stored, initialized to None.
        - coco_file: The COCO file associated with the dataset, initialized to None.

    Methods:
        - download_assets: Downloads the dataset assets to the local filesystem.
        - download_coco_file: Downloads the COCO file associated with the dataset to the local filesystem.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_version: DatasetVersion,
        destination_path: str,
        multi_asset: Optional[MultiAsset] = None,
        labelmap: Optional[dict[str, Label]] = None,
    ):
        """
        Initializes the DatasetContext with dataset metadata and configuration.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_version (DatasetVersion): The dataset version object.
            multi_asset (MultiAsset): The collection of assets for the dataset.
            labelmap (dict): The mapping of label names to ids.
            destination_path (str): The root directory for storing the dataset locally.
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.dataset_extraction_path = os.path.join(destination_path, self.dataset_name)
        os.makedirs(self.dataset_extraction_path, exist_ok=True)
        if not labelmap:
            self.labelmap = get_labelmap(dataset_version=dataset_version)
        else:
            self.labelmap = labelmap or {}
        if not multi_asset:
            try:
                self.multi_asset = dataset_version.list_assets()
            except NoDataError:
                self.multi_asset = None
        else:
            self.multi_asset = multi_asset
        self.image_dir = os.path.join(self.dataset_extraction_path, "images")
        self.coco_file = None

    def download_assets(self) -> None:
        """
        Downloads the dataset assets to a local directory. Initializes `image_dir`
        to the path where images are stored.
        """
        if self.multi_asset:
            self.multi_asset.download(target_path=self.image_dir, use_id=True)
        else:
            raise ValueError("No assets found in the dataset")

    def download_coco_file(self) -> None:
        """
        Builds the COCO file associated with the dataset. Initializes `coco_file` to the COCO file object.
        """
        if self.multi_asset:
            self.coco_file = self.dataset_version.build_coco_file_locally(
                assets=self.multi_asset,
                use_id=True,
            )
        else:
            raise ValueError("No assets found in the dataset")

    def download(self) -> None:
        self.download_assets()
        self.download_coco_file()
