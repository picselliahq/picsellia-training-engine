from src.models.model.common.model_context import ModelContext

import os
from typing import Optional, Dict, Any
from picsellia import ModelVersion, Label

import yaml


def find_latest_run_dir(dir):
    """
    Finds the latest run directory in the given directory.
    """
    run_dirs = os.listdir(dir)
    processed_run_dirs = {}

    for run_dir in run_dirs:
        run_id = -1
        if "-" in run_dir:
            try:
                run_id = int(run_dir.split("-")[1])
            except ValueError:
                pass

        while run_id in processed_run_dirs:
            run_id -= 1

        processed_run_dirs[run_id] = run_dir

    return processed_run_dirs[max(processed_run_dirs)]


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

    def set_hyperparameters_path(self, destination_path: str):
        """
        Downloads the hyperparameters file from Picsellia to the specified destination path.

        Args:
            destination_path (str): The directory path where the hyperparameters file will be saved.
        """
        hyperparameters_file = self.model_version.get_file(
            name=self.hyperparameters_name
        )
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

    def set_trained_weights_path(self):
        """
        Sets the path to the trained weights file using the latest run directory.
        """
        if not self.results_dir or not os.path.exists(self.results_dir):
            raise ValueError("The results directory is not set.")

        training_dir = os.path.join(self.results_dir, "training")
        latest_run = find_latest_run_dir(training_dir)

        print(f"latest_run: {latest_run}")

        trained_weights_dir = os.path.join(training_dir, latest_run, "weights")
        self.trained_weights_path = os.path.join(trained_weights_dir, "best.pt")

        print(f"trained_weights_path: {self.trained_weights_path}")
