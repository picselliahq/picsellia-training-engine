from src.models.model.common.model_context import ModelContext

import os
from typing import Optional, Dict, Any
from picsellia import ModelVersion, Label

import yaml


class Yolov7ModelContext(ModelContext):
    def __init__(
        self,
        model_name: str,
        model_version: ModelVersion,
        pretrained_weights_name: Optional[str] = None,
        trained_weights_name: Optional[str] = None,
        config_name: Optional[str] = None,
        exported_weights_name: Optional[str] = None,
        hyperparameters_name: Optional[str] = None,
        labelmap: Optional[Dict[str, Label]] = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            pretrained_weights_name=pretrained_weights_name,
            trained_weights_name=trained_weights_name,
            config_name=config_name,
            exported_weights_name=exported_weights_name,
            labelmap=labelmap,
        )
        self.hyperparameters_name = hyperparameters_name
        self.hyperparameters_path: Optional[str] = None

    def download_hyperparameters(self, destination_path: str):
        """
        Downloads the hyperparameters file from Picsellia to the specified destination path.

        Args:
            destination_path (str): The directory path where the hyperparameters file will be saved.
        """
        hyperparameters_file = self.model_version.get_file(
            name=self.hyperparameters_name
        )
        hyperparameters_file.download(target_path=destination_path)
        self.hyperparameters_path = os.path.join(
            destination_path, hyperparameters_file.filename
        )

    def update_hyperparameters(
        self, hyperparameters: Dict[str, Any], hyperparameters_path: str
    ):
        """
        Updates the hyperparameters with the provided dictionary.

        Args:
            hyperparameters (Dict[str, Any]): The dictionary of hyperparameters to update.
        """
        with open(hyperparameters_path, "w") as file:
            yaml.dump(hyperparameters, file)
