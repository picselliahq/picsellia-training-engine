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
    """Stores and manages the context of a dataset, including metadata, paths, and assets."""

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
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.destination_path = destination_path
        self.use_id = use_id
        self.download_annotations = download_annotations
        self.labelmap = labelmap or get_labelmap(dataset_version=dataset_version)
        self.multi_asset = multi_asset or (
            None if skip_asset_listing else self._list_assets()
        )

        self._initialize_paths()
        self.coco_file: COCOFile
        self._initialize_coco_file()

    def _initialize_paths(self):
        """Initializes the dataset, image, and annotations paths."""
        self.dataset_path = os.path.join(self.destination_path, self.dataset_name)
        self.image_dir = os.path.join(self.dataset_path, "images")
        self.annotations_dir = os.path.join(self.dataset_path, "annotations")

    def download_and_build_coco_file(self, annotations_dir: str) -> None:
        coco_file_path = self._download_coco_file(annotations_dir=annotations_dir)
        if isinstance(coco_file_path, str):
            self.coco_file_path = coco_file_path
            self.coco_file = self._build_coco_file(self.coco_file_path)
        else:
            logger.warning("COCO file path is None, skipping COCO file build.")

    def _download_coco_file(self, annotations_dir: str) -> Optional[str]:
        """Downloads the COCO file associated with the dataset."""
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
        """Builds the COCO file object and initializes the manager."""
        coco_file = read_coco_file(str(coco_file_path))
        self.coco_file_manager = COCOFileManager(coco_file)
        return coco_file

    def _initialize_coco_file(self):
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
        """Downloads the dataset assets to the local filesystem."""
        os.makedirs(image_dir, exist_ok=True)
        if self.multi_asset:
            self.multi_asset.download(target_path=str(image_dir), use_id=self.use_id)

    def _list_assets(self) -> Optional[MultiAsset]:
        """Lists the assets in the dataset."""
        try:
            return self.dataset_version.list_assets()
        except NoDataError:
            logger.warning(f"No assets found for dataset {self.dataset_name}")
            return None

    def load_coco_file_data(self) -> Dict[str, Any]:
        """
        Load COCO data from the annotation file.
        """
        with open(self.coco_file_path, "r") as f:
            return json.load(f)

    def get_assets_batch(self, limit: int, offset: int) -> MultiAsset:
        """Retrieves a batch of assets from the dataset."""
        return self.dataset_version.list_assets(limit=limit, offset=offset)


TDatasetContext = TypeVar("TDatasetContext", bound=DatasetContext)
