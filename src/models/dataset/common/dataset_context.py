import json
import os
from typing import Any, Dict, Optional
import logging
import os
from typing import Dict, Optional, TypeVar

from picsellia import DatasetVersion, Label
from picsellia.exceptions import NoDataError
from picsellia.sdk.asset import MultiAsset
from picsellia.types.enums import AnnotationFileType
from picsellia_annotations.coco import COCOFile
from picsellia_annotations.utils import read_coco_file

from src.models.dataset.common.coco_file_manager import COCOFileManager
from src.steps.data_extraction.utils.image_utils import get_labelmap

logger = logging.getLogger(__name__)


class DatasetContext:
    """
    Stores and manages the context of a dataset, including metadata, paths,
    assets, and COCO file management.

    Attributes:
        dataset_name (str): The name of the dataset.
        dataset_version (DatasetVersion): Version of the dataset from Picsellia.
        destination_path (str): Path where the dataset will be downloaded and stored.
        multi_asset (Optional[MultiAsset]): MultiAsset object for managing dataset assets.
        labelmap (Optional[Dict[str, Label]]): Label map of the dataset, used for annotations.
        skip_asset_listing (bool): If True, skips listing of assets during initialization.
        use_id (Optional[bool]): If True, uses asset ID in file paths.
        download_annotations (Optional[bool]): If True, downloads and initializes annotations.
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
        download_annotations: Optional[bool] = True,
    ):
        """
        Initializes the DatasetContext with the provided dataset information, path, and options.

        Args:
            dataset_name (str): Name of the dataset.
            dataset_version (DatasetVersion): Version of the dataset to manage.
            destination_path (str): Local path where dataset assets and annotations will be saved.
            multi_asset (Optional[MultiAsset]): Pre-listed assets to be used.
            labelmap (Optional[Dict[str, Label]]): Pre-loaded label map for the dataset.
            skip_asset_listing (bool): Whether to skip asset listing (default: False).
            use_id (Optional[bool]): Whether to use asset ID in file paths (default: True).
            download_annotations (Optional[bool]): Whether to download annotation files (default: True).
        """
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.destination_path = destination_path
        self.use_id = use_id
        self.download_annotations = download_annotations
        self.labelmap = labelmap or get_labelmap(dataset_version=dataset_version)
        if not labelmap:
            self.labelmap = get_labelmap(dataset_version=dataset_version)
        else:
            self.labelmap = labelmap or {}
        if multi_asset:
            self.multi_asset = multi_asset
        elif not skip_asset_listing:
            self.list_assets()

        self._initialize_paths()
        self.coco_file: COCOFile
        self._initialize_coco_file()

    def _initialize_paths(self):
        """
        Initializes the file paths for the dataset, images, and annotations.
        Sets up the necessary directories in the destination path.
        """
        self.dataset_path = os.path.join(self.destination_path, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_path, "images")
        self.annotations_dir = os.path.join(self.dataset_path, "annotations")

    def download_and_build_coco_file(self, annotations_dir: str) -> None:
        """
        Downloads the COCO annotation file and builds the COCO file object.

        Args:
            annotations_dir (str): Directory path to save the downloaded COCO file.
        """
        coco_file_path = self._download_coco_file(annotations_dir=annotations_dir)
        if isinstance(coco_file_path, str):
            self.coco_file_path = coco_file_path
            self.coco_file = self._build_coco_file(self.coco_file_path)
        else:
            logger.warning("COCO file path is None, skipping COCO file build.")

    def _download_coco_file(self, annotations_dir: str) -> Optional[str]:
        """
        Downloads the COCO file for the dataset from Picsellia.

        Args:
            annotations_dir (str): Path where the COCO file should be downloaded.

        Returns:
            Optional[str]: Path to the COCO file if downloaded successfully, otherwise None.
        """
        coco_file_path = self.dataset_version.export_annotation_file(
            annotation_file_type=AnnotationFileType.COCO,
            target_path=annotations_dir,
            assets=self.multi_asset,
            use_id=self.use_id,
        )
        if os.path.exists(coco_file_path):
            return coco_file_path
        else:
            logger.warning("COCO file download failed, no file found.")
            return None

    def _build_coco_file(self, coco_file_path: str) -> COCOFile:
        """
        Builds and returns a COCOFile object from the downloaded COCO file.

        Args:
            coco_file_path (str): Path to the downloaded COCO file.

        Returns:
            COCOFile: The COCO file object with the dataset's annotations.
        """
        coco_file = read_coco_file(str(coco_file_path))
        self.coco_file_manager = COCOFileManager(coco_file)
        return coco_file

    def _initialize_coco_file(self):
        """
        Initializes the COCO file by either downloading it or creating it locally
        depending on the configuration.
        Raises a ValueError if the COCO file cannot be downloaded.
        """
        if self.download_annotations:
            coco_file_path = self._download_coco_file(
                annotations_dir=str(self.annotations_dir)
            )
            if coco_file_path is None:
                raise ValueError(
                    "Failed to download COCO file. Dataset context cannot be initialized."
                )
            self.coco_file = self._build_coco_file(coco_file_path)
        else:
            if self.multi_asset:
                self.coco_file = self.dataset_version.build_coco_file_locally(
                    use_id=self.use_id
                )
            else:
                self.coco_file = COCOFile(images=[], annotations=[])
            self.coco_file_manager = COCOFileManager(self.coco_file)

    def download_assets(self, image_dir: str) -> None:
        """
        Downloads all assets (images) associated with the dataset.

        Args:
            image_dir (str): Directory where the assets should be downloaded.
        """
        os.makedirs(image_dir, exist_ok=True)
        if self.multi_asset:
            self.multi_asset.download(target_path=str(image_dir), use_id=self.use_id)

    def list_assets(self) -> None:
        """
        Lists the assets in the dataset.

        Raises:
            NoDataError: If no assets are found in the dataset.
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
        Retrieves a batch of assets with a specified limit and offset.

        Args:
            limit (int): Number of assets to retrieve in one batch.
            offset (int): Starting point for the asset batch.

        Returns:
            MultiAsset: Batch of assets.
        """
        return self.dataset_version.list_assets(limit=limit, offset=offset)


TDatasetContext = TypeVar("TDatasetContext", bound=DatasetContext)
