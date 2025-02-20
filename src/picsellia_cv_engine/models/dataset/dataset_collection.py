import os
from typing import Generic, List, Optional, Iterator

from src.picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)

import logging

logger = logging.getLogger(__name__)


class DatasetCollection(Generic[TBaseDatasetContext]):
    """
    A collection of dataset contexts for different splits of a dataset.

    This class aggregates dataset contexts for the common splits used in machine learning projects:
    training, validation, and testing. It provides a convenient way to access and manipulate these
    dataset contexts as a unified object. The class supports direct access to individual dataset
    contexts, iteration over all contexts, and collective operations on all contexts, such as downloading
    assets.

    Attributes:
        datasets (dict): A dictionary containing dataset contexts, where keys are dataset names.
        dataset_path (Optional[str]): The common file path for all dataset splits. Initialized after calling `download_all`.
    """

    def __init__(self, datasets: List[TBaseDatasetContext]):
        """
        Initializes the collection with a list of dataset contexts.

        Args:
            datasets (List[TDatasetContext]): A list of dataset contexts for different splits (train, val, test).
        """
        self.datasets = {dataset.dataset_name: dataset for dataset in datasets}
        self.dataset_path: Optional[str] = None

    def __getitem__(self, key: str) -> TBaseDatasetContext:
        """
        Retrieves a dataset context by its name.

        Args:
            key (str): The name of the dataset context.

        Returns:
            TDatasetContext: The dataset context corresponding to the given name.

        Raises:
            KeyError: If the provided key does not exist in the collection.
        """
        return self.datasets[key]

    def __setitem__(self, key: str, value: TBaseDatasetContext):
        """
        Sets or updates a dataset context in the collection.

        Args:
            key (str): The name of the dataset context to update or add.
            value (TDatasetContext): The dataset context object to associate with the given name.
        """
        self.datasets[key] = value

    def __iter__(self) -> Iterator[TBaseDatasetContext]:
        """
        Iterates over all dataset contexts in the collection.

        Returns:
            Iterator[TDatasetContext]: An iterator over the dataset contexts.
        """
        return iter(self.datasets.values())

    def download_all(
        self,
        images_destination_path: str,
        annotations_destination_path: str,
        use_id: Optional[bool] = True,
        skip_asset_listing: Optional[bool] = False,
    ) -> None:
        """
        Downloads all assets and annotations for every dataset context in the collection.

        For each dataset context, this method:
        1. Downloads the assets (images) to the corresponding image directory.
        2. Downloads and builds the COCO annotation file for each dataset.

        Args:
            destination_path (str): The directory where all datasets (images and annotations) will be saved.
            use_id (Optional[bool]): Whether to use asset IDs in the file paths. If None, the internal logic of each dataset context will handle it.
            skip_asset_listing (bool, optional): If True, skips listing the assets when downloading. Defaults to False.

        Example:
            If you want to download assets and annotations for both train and validation datasets,
            this method will create two directories (e.g., `train/images`, `train/annotations`,
            `val/images`, `val/annotations`) under the specified `destination_path`.
        """
        for dataset_context in self:
            logger.info(f"Downloading assets for {dataset_context.dataset_name}")
            dataset_context.download_assets(
                destination_dir=os.path.join(
                    images_destination_path, dataset_context.dataset_name
                ),
                use_id=use_id,
                skip_asset_listing=skip_asset_listing,
            )

            logger.info(f"Downloading annotations for {dataset_context.dataset_name}")
            dataset_context.download_annotations(
                destination_path=os.path.join(
                    annotations_destination_path, dataset_context.dataset_name
                ),
                use_id=use_id,
            )
