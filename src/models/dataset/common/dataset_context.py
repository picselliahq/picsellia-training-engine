import json
from typing import Any
import logging
import os
from typing import Dict, Optional, TypeVar

from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset
from picsellia.types.enums import AnnotationFileType
from picsellia_annotations.coco import COCOFile
from picsellia_annotations.utils import read_coco_file

from src.steps.data_extraction.utils.image_utils import get_labelmap

logger = logging.getLogger(__name__)


class DatasetContext:
    """
    A class to store and manage the context of a dataset, including metadata, paths,
    assets, and COCO file management.

    Attributes:
        dataset_name (str): The name of the dataset.
        dataset_version (DatasetVersion): The version of the dataset from Picsellia.
        assets (Optional[MultiAsset]): Optional object for managing dataset assets.
        labelmap (Optional[Dict[str, Label]]): A map of dataset labels used for annotations.
        images_dir (Optional[str]): Directory where image assets are downloaded.
        annotations_dir (Optional[str]): Directory where annotation files are stored.
        coco_file_path (Optional[str]): Path to the COCO annotation file.
        coco_file (Optional[COCOFile]): COCOFile object to handle dataset annotations.
    """

    def __init__(
        self,
        dataset_name: str,
        dataset_version: DatasetVersion,
        assets: Optional[MultiAsset] = None,
        labelmap: Optional[Dict[str, Label]] = None,
    ):
        """
        Initializes the DatasetContext with the given dataset name, version, assets, and labelmap.

        Args:
            dataset_name (str): The name of the dataset.
            dataset_version (DatasetVersion): The dataset version from Picsellia.
            assets (Optional[MultiAsset]): Optional assets object. If not provided, assets will be managed automatically.
            labelmap (Optional[Dict[str, Label]]): Pre-loaded label map for the dataset. If not provided, the label map will be fetched.
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.assets = assets

        if not labelmap:
            self.labelmap = get_labelmap(dataset_version=dataset_version)
        else:
            self.labelmap = labelmap or {}

        self.images_dir: Optional[str] = None
        self.annotations_dir: Optional[str] = None
        self.coco_file_path: Optional[str] = None
        self.coco_file: Optional[COCOFile] = None

    def download_assets(
        self,
        destination_path: str,
        use_id: Optional[bool] = True,
        skip_asset_listing: bool = False,
    ) -> None:
        """
        Downloads all assets (e.g., images) associated with the dataset to the specified directory.

        Args:
            destination_path (str): Directory where the assets will be saved.
            use_id (Optional[bool]): If True, uses asset IDs when creating file paths.
            skip_asset_listing (bool, optional): If True, skips the asset listing after downloading. Defaults to False.
        """
        os.makedirs(destination_path, exist_ok=True)
        if self.assets:
            self.assets.download(target_path=str(destination_path), use_id=use_id)
        else:
            self.dataset_version.download(
                target_path=str(destination_path), use_id=use_id
            )
            if not skip_asset_listing:
                self.assets = self.dataset_version.list_assets()
        self.images_dir = destination_path

    def download_and_build_coco_file(
        self,
        destination_path: str,
        use_id: Optional[bool] = True,
        skip_asset_listing: Optional[bool] = False,
    ) -> None:
        """
        Downloads the COCO annotation file and builds the corresponding COCOFile object.

        Args:
            destination_path (str): The directory path where the COCO file will be saved.
            use_id (Optional[bool]): If True, uses the asset ID when creating file paths.
            skip_asset_listing (bool, optional): If True, skips the asset listing after downloading. Defaults to False.
        """
        coco_file_path = self._download_coco_file(
            destination_path=destination_path,
            use_id=use_id,
            skip_asset_listing=skip_asset_listing,
        )
        if isinstance(coco_file_path, str):
            self.annotations_dir = destination_path
            self.coco_file_path = coco_file_path
            self.coco_file = self._build_coco_file(coco_file_path=self.coco_file_path)
        else:
            logger.warning("COCO file path is None, skipping COCO file build.")

    def load_coco_file_data(self) -> Dict[str, Any]:
        """
        Loads the COCO data from the annotation file.

        Returns:
            Dict[str, Any]: The loaded COCO data as a dictionary.
        """
        if self.coco_file_path is None:
            raise FileNotFoundError(
                "COCO file path is not set. Please download the COCO file first."
            )
        with open(self.coco_file_path, "r") as f:
            return json.load(f)

    def get_assets_batch(self, limit: int, offset: int) -> MultiAsset:
        """
        Retrieves a batch of assets from the dataset with a specified limit and offset.

        Args:
            limit (int): The number of assets to retrieve in the batch.
            offset (int): The starting point for asset retrieval.

        Returns:
            MultiAsset: A batch of assets from the dataset.
        """
        return self.dataset_version.list_assets(limit=limit, offset=offset)

    def _download_coco_file(
        self,
        destination_path: str,
        use_id: Optional[bool] = True,
        skip_asset_listing: Optional[bool] = False,
    ) -> Optional[str]:
        """
        Downloads the COCO annotation file for the dataset from Picsellia.

        Args:
            destination_path (str): Path where the COCO file will be saved.
            use_id (bool): If True, uses asset IDs when generating the file path.
            skip_asset_listing (bool, optional): If True, skips the asset listing after downloading. Defaults to False.

        Returns:
            Optional[str]: The file path to the downloaded COCO file, or None if the download fails.
        """
        if self.assets:
            coco_file_path = self.dataset_version.export_annotation_file(
                annotation_file_type=AnnotationFileType.COCO,
                target_path=destination_path,
                assets=self.assets,
                use_id=use_id,
            )
        else:
            coco_file_path = self.dataset_version.export_annotation_file(
                annotation_file_type=AnnotationFileType.COCO,
                target_path=destination_path,
                use_id=use_id,
            )
            if not skip_asset_listing:
                self.assets = self.dataset_version.list_assets()

        if os.path.exists(coco_file_path):
            return coco_file_path
        else:
            logger.warning("COCO file download failed, no file found.")
            return None

    def _build_coco_file(self, coco_file_path: str) -> COCOFile:
        """
        Builds and returns a COCOFile object from the downloaded COCO annotation file.

        Args:
            coco_file_path (str): Path to the downloaded COCO file.

        Returns:
            COCOFile: The COCOFile object that handles the dataset's annotations.
        """
        coco_file = read_coco_file(str(coco_file_path))
        return coco_file


TDatasetContext = TypeVar("TDatasetContext", bound=DatasetContext)
