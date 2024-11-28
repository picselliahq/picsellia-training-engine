import logging
import os
import re
from abc import abstractmethod
from pathlib import Path
from typing import List, Union

from picsellia import Experiment, ModelFile, ModelVersion

from src.models.model.common.model_context import ModelContext

logger = logging.getLogger("picsellia-engine")


class ModelContextExporter:
    """
    Base class for exporting and saving a model context.

    This class serves as a base for exporting a model and saving it to an experiment.
    It provides an abstract method `export_model_context` for subclasses to implement
    specific export logic, and a concrete method `save_model_to_experiment` for saving
    the exported model to the experiment.

    Attributes:
        model_context (ModelContext): The context of the model to be exported.
    """

    def __init__(self, model_context: ModelContext):
        """
        Initializes the ModelContextExporter with the given model context and experiment.

        Args:
            model_context (ModelContext): The model context containing the model's details.
        """
        self.model_context = model_context

    @abstractmethod
    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ):
        """
        Abstract method to export the model context.

        This method should be implemented by subclasses to define the logic for exporting
        the model context in the specified format.

        Args:
            exported_model_destination_path (str): The destination path where the exported model will be saved.
            export_format (str): The format in which the model should be exported.
        """
        pass

    def save_model_to_experiment(
        self,
        experiment: Experiment,
        exported_weights_path: str,
        exported_weights_name: str,
    ) -> None:
        """
        Saves the exported model to the specified experiment.

        This method takes the directory where the model was exported and uploads it to the
        associated experiment. If multiple files exist in the directory, they are zipped before uploading.

        Args:
            experiment (Experiment): The experiment to which the model should be saved.
            exported_weights_path (str):  The path where the exported model weights are stored.
            exported_weights_name (str): The name under which the model will be stored in the experiment.
        """
        self._store_artifact(
            target=experiment,
            exported_weights_path=exported_weights_path,
            exported_weights_name=exported_weights_name,
        )

    def save_model_to_model_version(
        self,
        model_version: ModelVersion,
        exported_weights_path: str,
        exported_weights_name: str,
    ) -> None:
        """
        Saves the exported model to the specified model version.

        This method takes the directory where the model was exported and uploads it to the
        associated model version. If multiple files exist in the directory, they are zipped before uploading.

        Args:
            model_version (Experiment): The model version to which the model should be saved.
            exported_weights_path (str): The path where the exported model weights are stored.
            exported_weights_name (str): The name under which the model will be stored in the experiment.
        """
        self._store_artifact(
            target=model_version,
            exported_weights_path=exported_weights_path,
            exported_weights_name=exported_weights_name,
        )

    def _get_unique_file_name(
        self, exported_weights_name: str, target_files: List[ModelFile]
    ) -> str:
        """
        Get a unique filename for the exported model by looking if a file with the same name already exists in the
        target. If so, append a number to the filename to make it unique.
        Args:
            exported_weights_name:
            target_files:

        Returns:

        """
        unique_name = self._sanitize_filename(filename=exported_weights_name)
        existing_files = [file.name for file in target_files]

        if unique_name in existing_files:
            i = 2

            while f"{unique_name}_{i}" in existing_files:
                i += 1

            unique_name = f"{unique_name}_{i}"
            logger.warning(
                f"⚠️ Model with name {exported_weights_name} already exists in the target. Renaming to {unique_name}"
            )

        return unique_name

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to comply with Picsellia naming requirements.

        Args:
            filename: Original filename to sanitize

        Returns:
            Sanitized filename containing only ascii chars, numbers, underscores and hyphens

        Example:
            >>> sanitize_filename("model v1.0 (best)!")
            >>> "model_v1_0_best"
        """
        # Replace spaces and invalid chars with underscore
        sanitized = re.sub(r"[^a-zA-Z0-9-]", "_", filename)

        # Remove multiple consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)

        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        return sanitized

    def _store_artifact(
        self,
        target: Union[Experiment, ModelVersion],
        exported_weights_path: str,
        exported_weights_name: str,
    ) -> None:
        """
        Stores the exported model weights to the specified target.

        Args:
            target: The target to which the model weights should be stored.
            exported_weights_path: The path where the exported model weights are stored.
            exported_weights_name: The name under which the model will be stored in the target.

        Raises:
            ValueError: If no model files are found in the exported model directory.
        """
        weights_dir = Path(exported_weights_path)
        if not weights_dir.exists():
            raise ValueError(f"Export directory does not exist: {weights_dir}")

        exported_files = list(weights_dir.iterdir())
        if not exported_files:
            raise ValueError(f"No model files found in: {weights_dir}")

        exported_weights_name = self._get_unique_file_name(
            exported_weights_name, target.list_files()
        )

        if len(exported_files) > 1 or weights_dir.is_dir():
            target.store(
                name=exported_weights_name,
                path=exported_weights_path,
                do_zip=True,
            )
        else:
            target.store(
                name=exported_weights_name,
                path=os.path.join(
                    exported_weights_path,
                    exported_files[0],
                ),
            )
