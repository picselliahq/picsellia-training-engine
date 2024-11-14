from typing import Optional, List
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import TDatasetContext

import os
import yaml


class Yolov7DatasetCollection(DatasetCollection):
    def __init__(self, datasets: List[TDatasetContext]):
        super().__init__(datasets=datasets)
        self.config_path: Optional[str] = None

    def download_all(
        self,
        destination_path: str,
        use_id: Optional[bool] = None,
        skip_asset_listing: bool = False,
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
            # Download dataset assets (images) into 'images' directory
            print(f"Downloading assets for {dataset_context.dataset_name}")
            dataset_context.download_assets(
                destination_path=os.path.join(
                    destination_path, dataset_context.dataset_name, "images"
                ),
                use_id=use_id,
                skip_asset_listing=skip_asset_listing,
            )

        # Set the common dataset path for all splits
        self.dataset_path = destination_path

    def write_config(self, config_path: str) -> None:
        """
        Writes the dataset collection configuration to a YAML file.

        Args:
            config_path (str): The path to the configuration file.
        """
        if not self.dataset_path:
            raise ValueError(
                "Dataset path is required to write the configuration file."
            )
        with open(config_path, "w") as f:
            data = {
                "train": os.path.join(self.dataset_path, "images", "train"),
                "val": os.path.join(self.dataset_path, "images", "val"),
                "test": os.path.join(self.dataset_path, "images", "test"),
                "nc": len(self["train"].labelmap),
                "names": list(self["train"].labelmap.keys()),
            }
            yaml.dump(data, f)
        self.config_path = config_path
