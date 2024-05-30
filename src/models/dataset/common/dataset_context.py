import json
import os
from typing import Any, Dict, Optional

from picsellia import DatasetVersion, Label
from picsellia.exceptions import NoDataError
from picsellia.sdk.asset import MultiAsset
from picsellia.types.enums import AnnotationFileType
from picsellia_annotations.coco import COCOFile
from picsellia_annotations.utils import read_coco_file

from src.steps.data_extraction.utils.image_utils import get_labelmap


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
        labelmap: Optional[Dict[str, Label]] = None,
        skip_asset_listing: bool = False,
        use_id: Optional[bool] = True,
    ):
        """
        Initializes the DatasetContext with dataset metadata and configuration.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_version (DatasetVersion): The dataset version object.
            multi_asset (MultiAsset): The collection of assets for the dataset.
            labelmap (dict): The mapping of label names to ids.
            destination_path (str): The root directory for storing the dataset locally.
            skip_asset_listing (bool): Whether to skip listing the dataset's assets.
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.destination_path = destination_path
        self.use_id = use_id
        if not labelmap:
            self.labelmap = get_labelmap(dataset_version=dataset_version)
        else:
            self.labelmap = labelmap or {}

        if multi_asset:
            self.multi_asset = multi_asset
        elif not skip_asset_listing:
            self.list_assets()
    def build_coco_file(self) -> COCOFile:
        """
        Builds the COCO file associated with the dataset. Initializes `coco_file` to the COCO file object.
        """
        if self.multi_asset:
            return self.dataset_version.build_coco_file_locally(
                assets=self.multi_asset,
                use_id=self.use_id,
            )
        else:
            self.multi_asset = None
        self.image_dir = os.path.join(destination_path, self.dataset_name, "images")
        self.coco_file_path = self.download_coco_file()
        self.coco_file = self.build_coco_file(coco_file_path=self.coco_file_path)

    def download_coco_file(self) -> str:
        """
        Downloads the COCO file associated with the dataset to the local filesystem.
        """
        return self.dataset_version.export_annotation_file(
            annotation_file_type=AnnotationFileType.COCO,
            target_path=os.path.join(
                self.destination_path, self.dataset_name, "annotations"
            ),
            assets=self.multi_asset,
            use_id=self.use_id,
        )

    def build_coco_file(self, coco_file_path) -> COCOFile:
        """
        Builds the COCO file object associated with the dataset.

        Returns:
            - COCOFile: The COCO file object.
        """
        return read_coco_file(coco_file_path)

    def download_assets(self) -> None:
        """
        Downloads the dataset assets to a local directory. Initializes `image_dir`
        to the path where images are stored.
        """
        os.makedirs(self.image_dir, exist_ok=True)
        if self.multi_asset:
            os.makedirs(self.image_dir, exist_ok=True)
            self.multi_asset.download(target_path=self.image_dir, use_id=self.use_id)
        else:
            print(f"No assets found for dataset {self.dataset_name}!")

    def list_assets(self) -> None:
        """
        Lists the assets in the dataset.
        """
        try:
            self.multi_asset = self.dataset_version.list_assets()
        except NoDataError:
            self.multi_asset = None

    def load_coco_file_data(self) -> Dict[str, Any]:
        """
        Load COCO data from the annotation file.
        """
        with open(self.coco_file_path, "r") as f:
            return json.load(f)

    def get_assets_batch(self, limit: int, offset: int) -> MultiAsset:
        """
        Lists the assets in the dataset in batches.

        Args:
            limit: The maximum number of assets to retrieve.
            offset: The offset from which to start retrieving assets.

        Returns:
            A MultiAsset object containing the assets retrieved, limited by the `limit` and `offset` parameters.
        """
        return self.dataset_version.list_assets(limit=limit, offset=offset)
