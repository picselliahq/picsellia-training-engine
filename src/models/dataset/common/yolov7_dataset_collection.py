from typing import Optional, List
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import TDatasetContext

import os
import yaml


class Yolov7DatasetCollection(DatasetCollection):
    def __init__(self, datasets: List[TDatasetContext]):
        super().__init__(datasets=datasets)
        self.config_path: Optional[str] = None

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
