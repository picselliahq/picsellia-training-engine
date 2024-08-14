import json
import os
from typing import Any, Dict, Optional
import logging
import shutil
from pathlib import Path
from typing import Dict, Optional

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
        use_id: bool = True,
    ):
        self.dataset_name = dataset_name
        self.dataset_version = dataset_version
        self.destination_path = Path(destination_path)
        self.use_id = use_id
        self.labelmap = labelmap or get_labelmap(dataset_version=dataset_version)
        self.multi_asset = multi_asset or (
            None if skip_asset_listing else self._list_assets()
        )

        self._initialize_paths()
        self.coco_file_path: Optional[Path] = None
        self.coco_file: Optional[COCOFile] = None

    def _initialize_paths(self):
        """Initializes the dataset, image, and annotations paths."""
        self.dataset_path = self.destination_path / self.dataset_name
        self.image_dir = self.dataset_path / "images"
        self.annotations_dir = self.dataset_path / "annotations"

    def download_and_build_coco_file(self):
        coco_file_path = self._download_coco_file()
        if isinstance(coco_file_path, Path):
            self.coco_file_path = coco_file_path
            self.coco_file = self._build_coco_file(self.coco_file_path)
        else:
            logger.warning("COCO file path is None, skipping COCO file build.")

    def _download_coco_file(self) -> Optional[Path]:
        """Downloads the COCO file associated with the dataset."""
        coco_file_path = Path(
            self.dataset_version.export_annotation_file(
                annotation_file_type=AnnotationFileType.COCO,
                target_path=str(self.annotations_dir),
                assets=self.multi_asset,
                use_id=self.use_id,
            )
        )
        if coco_file_path.exists():
            return coco_file_path
        else:
            logger.warning("COCO file download failed, no file found.")
            return None

    def _build_coco_file(self, coco_file_path: Path) -> COCOFile:
        """Builds the COCO file object and initializes the manager."""
        coco_file = read_coco_file(str(coco_file_path))
        self.coco_file_manager = COCOFileManager(coco_file)
        return coco_file

    def download_assets(self) -> None:
        """Downloads the dataset assets to the local filesystem."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        if self.multi_asset:
            self.multi_asset.download(
                target_path=str(self.image_dir), use_id=self.use_id
            )
        else:
            logger.warning(f"No assets found for dataset {self.dataset_name}!")

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

    def update_destination_path(self, new_path: str) -> Dict[str, bool]:
        """Updates the destination path and verifies dataset integrity."""
        self.destination_path = Path(new_path)
        self._initialize_paths()
        self.update_image_dir(self.image_dir)
        self.update_annotations_dir(self.annotations_dir)

        return self._verify_dataset_integrity()

    def update_image_dir(self, new_image_dir: Path) -> Dict[str, bool]:
        """Updates the image directory and verifies dataset integrity."""
        old_image_dir = self.image_dir
        self.image_dir = Path(new_image_dir)
        self._move_directory_if_empty(old_image_dir, self.image_dir)

        return self._verify_dataset_integrity()

    def update_annotations_dir(self, new_annotations_dir: Path) -> Dict[str, bool]:
        """Updates the annotations directory and verifies dataset integrity."""
        old_annotations_dir = self.annotations_dir
        self.annotations_dir = Path(new_annotations_dir)
        self._move_directory_if_empty(old_annotations_dir, self.annotations_dir)

        # Update COCO file path after moving the annotations directory
        if self.coco_file_path:
            self.coco_file_path = self.annotations_dir / self.coco_file_path.name
            self._update_coco_file(self.coco_file_path)

        return self._verify_dataset_integrity()

    def _move_directory_if_empty(self, old_dir: Path, new_dir: Path):
        """Moves files to the new directory only if it is empty, otherwise just removes the old directory."""
        new_dir.mkdir(parents=True, exist_ok=True)
        if not list(new_dir.iterdir()):  # Check if the new directory is empty
            if old_dir.exists():
                for file in old_dir.iterdir():
                    file.rename(new_dir / file.name)
        else:
            logger.info(
                f"The new directory {new_dir} already contains files. Skipping move."
            )

        # Remove old directory if it's empty
        if old_dir.exists():
            shutil.rmtree(old_dir)

    def _update_coco_file(self, new_coco_file_path: Optional[Path]):
        if new_coco_file_path and new_coco_file_path.exists():
            self.coco_file_path = new_coco_file_path
            self.coco_file = self._build_coco_file(self.coco_file_path)
        else:
            logger.warning(
                f"COCO file not found or path is None at {new_coco_file_path}"
            )
            self.coco_file = None
            self.coco_file_path = None

    def _verify_dataset_integrity(self) -> Dict[str, bool]:
        """Verifies the integrity of the dataset."""
        integrity = {
            "images_exist": self._verify_images_exist(),
        }
        logger.info(f"Dataset integrity check: {integrity}")
        return integrity

    def _verify_images_exist(self) -> bool:
        """Verifies if images exist in the image directory."""
        if not self.image_dir.exists():
            return False
        image_files = [
            f
            for f in self.image_dir.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]
        return len(image_files) > 0
