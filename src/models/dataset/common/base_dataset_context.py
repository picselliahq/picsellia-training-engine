from abc import abstractmethod
import logging
import os
from typing import Dict, Optional, TypeVar

from picsellia import DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset
from picsellia.exceptions import NoDataError

from src.models.utils.dataset_logging import get_labelmap

logger = logging.getLogger(__name__)


class BaseDatasetContext:
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

    def download_assets(
        self,
        destination_path: str,
        use_id: Optional[bool] = True,
        skip_asset_listing: Optional[bool] = False,
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
            try:
                self.dataset_version.download(
                    target_path=str(destination_path), use_id=use_id
                )
            except NoDataError:
                logger.warning(
                    "No assets found in the dataset version, skipping asset download."
                )
            if not skip_asset_listing:
                try:
                    self.assets = self.dataset_version.list_assets()
                except NoDataError:
                    logger.warning(
                        "No assets found in the dataset version, skipping asset listing."
                    )
        self.images_dir = destination_path

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

    @abstractmethod
    def download_annotations(
        self, destination_path: str, use_id: Optional[bool] = True
    ) -> None:
        """
        Downloads the annotations for the dataset to the specified directory.
        """
        pass


TBaseDatasetContext = TypeVar("TBaseDatasetContext", bound=BaseDatasetContext)
