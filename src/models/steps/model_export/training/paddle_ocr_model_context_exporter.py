import logging
import os
import subprocess
from typing import Union

import yaml
from picsellia import Experiment

from src.models.model.common.model_context import ModelContext
from src.models.steps.model_export.common.model_context_exporter import (
    ModelContextExporter,
)

logger = logging.getLogger(__name__)


class PaddleOCRModelContextExporter(ModelContextExporter):
    """
    Handles the export of a PaddleOCR model by preparing the configuration, finding the trained model,
    and executing the export process.

    This class extends the `ModelContextExporter` to implement export logic specific to PaddleOCR models.

    Attributes:
        model_context (ModelContext): The context containing model configuration and paths.
        experiment (Experiment): The Picsellia experiment associated with the model.
        config (dict): The configuration loaded from the model context's config file.
        current_pythonpath (str): The current PYTHONPATH environment variable before modifications.
    """

    def __init__(self, model_context: ModelContext, experiment: Experiment):
        """
        Initializes the PaddleOCRModelContextExporter with the provided model context and experiment.

        Args:
            model_context (ModelContext): The context containing the PaddleOCR model information.
            experiment (Experiment): The Picsellia experiment associated with the model export.
        """
        super().__init__(model_context=model_context, experiment=experiment)
        self.config = self.get_config()
        self.current_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = f".:{self.current_pythonpath}"

    def get_config(self) -> dict:
        """
        Loads the configuration file from the model context.

        Returns:
            dict: The loaded configuration file as a dictionary.

        Raises:
            ValueError: If no configuration file path is found in the model context.
        """
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        with open(self.model_context.config_path, "r") as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        return config

    def write_config(self):
        """
        Writes the current configuration back to the model context's configuration file.

        Raises:
            ValueError: If no configuration file path is found in the model context.
        """
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        with open(self.model_context.config_path, "w") as file:
            yaml.dump(self.config, file)

    def find_model_path(self, saved_model_path: str) -> Union[str, None]:
        """
        Finds the best or latest trained model file in the provided directory.

        Args:
            saved_model_path (str): The path to the directory containing saved model files.

        Returns:
            Union[str, None]: The path to the best or latest model if found, otherwise None.
        """
        model_files = [
            f
            for f in os.listdir(saved_model_path)
            if os.path.isfile(os.path.join(saved_model_path, f))
        ]
        for model_file in model_files:
            if isinstance(model_file, str):
                if model_file.startswith("best_accuracy"):
                    return os.path.join(saved_model_path, "best_accuracy")
                if model_file.startswith("latest"):
                    return os.path.join(saved_model_path, "latest")
        return None

    def export_model(self):
        """
        Executes the PaddleOCR model export process by running the export script.

        The method prepares the Python environment and runs the PaddleOCR export command.
        Logs any errors encountered during the export process.

        Raises:
            ValueError: If no configuration file path is found in the model context.
        """
        if not self.model_context.config_path:
            raise ValueError("No configuration file path found in model context")
        command = [
            "python3.10",
            "src/pipelines/paddle_ocr/PaddleOCR/tools/export_model.py",
            "-c",
            self.model_context.config_path,
        ]

        os.setuid(os.geteuid())

        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        process.wait()
        if process.returncode != 0:
            logger.error("Export failed with errors")
            if process.stderr:
                errors = process.stderr.read()
                logger.error(errors)

        # Restore original PYTHONPATH
        os.environ["PYTHONPATH"] = self.current_pythonpath

    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ):
        """
        Prepares and exports the model context by finding the trained model, updating the config, and running the export process.

        Args:
            exported_model_destination_path (str): The path where the exported model will be saved.
            export_format (str): The format in which the model will be exported (e.g., 'onnx', 'paddle').

        Raises:
            ValueError: If no trained weights directory is found in the model context or if no model files are found after export.
        """
        saved_model_dir = self.model_context.trained_weights_dir
        if not saved_model_dir:
            raise ValueError("No trained weights directory found in model context")

        found_model_path = self.find_model_path(saved_model_dir)

        if not found_model_path:
            logger.info(f"No model found in {saved_model_dir}, skipping export...")
        else:
            # Update the config with model paths and save inference directory
            self.config["Global"]["pretrained_model"] = found_model_path
            self.config["Global"][
                "save_inference_dir"
            ] = exported_model_destination_path
            self.write_config()

            # Run the export process
            self.export_model()

        # Check if model files were exported
        exported_model = os.listdir(exported_model_destination_path)
        if not exported_model:
            raise ValueError("No model files found in the exported model directory")
        else:
            logger.info(f"Model exported to {exported_model_destination_path}")
