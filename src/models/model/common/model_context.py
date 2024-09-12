import os
from typing import Optional, Dict, Any, TypeVar

from picsellia import ModelVersion, Label

from src.models.model.common.model_downloader import ModelDownloader


class ModelContext:
    """
    Manages the context of a specific model version, including paths, weights, configuration, and labels.

    This class handles the organization of model-related files, such as pretrained weights, trained weights,
    configuration files, and exported weights. It also provides functionality for downloading these files
    and managing the model's runtime instance.

    Attributes:
        model_name (str): The name of the model.
        model_version (ModelVersion): The version of the model from Picsellia.
        destination_path (str): The base directory where model files will be stored.
        pretrained_weights_name (Optional[str]): The name of the pretrained weights file attached to the model version in Picsellia.
        trained_weights_name (Optional[str]): The name of the trained weights file attached to the model version in Picsellia.
        config_name (Optional[str]): The name of the configuration file attached to the model version in Picsellia.
        exported_weights_name (Optional[str]): The name of the exported weights file attached to the model version in Picsellia.
        labelmap (Optional[Dict[str, Label]]): A dictionary mapping category names to labels.
        prefix_model_name (Optional[str]): A prefix for handling multiple models (e.g., OCR models with different purposes).
    """

    def __init__(
        self,
        model_name: str,
        model_version: ModelVersion,
        destination_path: str,
        pretrained_weights_name: Optional[str] = None,
        trained_weights_name: Optional[str] = None,
        config_name: Optional[str] = None,
        exported_weights_name: Optional[str] = None,
        labelmap: Optional[Dict[str, Label]] = None,
        prefix_model_name: Optional[str] = None,
    ):
        """
        Initializes the ModelContext, which manages the paths, version, and related information for a specific model.

        Args:
            model_name (str): The name of the model.
            model_version (ModelVersion): The version of the model, which contains the pretrained model and configuration.
            destination_path (str): The base directory where model files will be stored.
            pretrained_weights_name (Optional[str], optional): The name of the pretrained weights file attached to the model version in Picsellia. Defaults to None.
            trained_weights_name (Optional[str], optional): The name of the trained weights file attached to the model version in Picsellia. Defaults to None.
            config_name (Optional[str], optional): The name of the configuration file attached to the model version in Picsellia. Defaults to None.
            exported_weights_name (Optional[str], optional): The name of the exported weights file attached to the model version in Picsellia. Defaults to None.
            labelmap (Optional[Dict[str, Label]], optional): A dictionary mapping category names to labels. Defaults to None.
            prefix_model_name (Optional[str], optional): A prefix used when the model version includes multiple models (e.g., in OCR models, one for bounding boxes and one for text: these models will be prefixed by terms like "bbox-" or "text-"). Defaults to None.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.destination_path = destination_path

        self.pretrained_weights_name = pretrained_weights_name
        self.trained_weights_name = trained_weights_name
        self.config_name = config_name
        self.exported_weights_name = exported_weights_name
        self.prefix_model_name = prefix_model_name

        self.labelmap = labelmap or {}

        self.weights_dir = self._create_directory("weights")
        self.results_dir = self._create_directory("results")

        self.pretrained_weights_dir: Optional[str] = None
        self.trained_weights_dir: Optional[str] = None
        self.config_dir: Optional[str] = None
        self.exported_weights_dir: Optional[str] = None

        self.pretrained_weights_path: Optional[str] = None
        self.trained_weights_path: Optional[str] = None
        self.config_path: Optional[str] = None
        self.exported_weights_path: Optional[str] = None

        self._loaded_model: Optional[Any] = None

    def _create_directory(self, folder_name: str) -> str:
        """
        Creates a directory inside the destination path for a specific type of model-related files (e.g., weights, results).

        Args:
            folder_name (str): The name of the folder to create (e.g., 'weights', 'results').

        Returns:
            str: The path to the created directory.
        """
        if self.prefix_model_name:
            path = os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                folder_name,
            )
        else:
            path = os.path.join(self.destination_path, self.model_name, folder_name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def loaded_model(self) -> Any:
        """
        Returns the loaded model instance. Raises an error if the model is not yet loaded.

        Returns:
            Any: The loaded model instance.

        Raises:
            ValueError: If the model is not loaded, an error is raised.
        """
        if self._loaded_model is None:
            raise ValueError(
                "Model is not loaded. Please load the model before accessing it."
            )
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """
        Sets the provided model instance as the loaded model.

        Args:
            model (Any): The model instance to set as loaded.
        """
        self._loaded_model = model

    def download_weights(self, model_weights_destination_path: str) -> None:
        """
        Downloads the model's weights and configuration files to the specified destination path.

        Args:
            model_weights_destination_path (str): The destination path where the model weights and related files will be downloaded.
        """
        downloader = ModelDownloader()

        # Create directories
        self.weights_dir = model_weights_destination_path
        self.pretrained_weights_dir = os.path.join(
            model_weights_destination_path, "pretrained_weights"
        )
        self.trained_weights_dir = os.path.join(
            model_weights_destination_path, "trained_weights"
        )
        self.config_dir = os.path.join(model_weights_destination_path, "config")
        self.exported_weights_dir = os.path.join(
            model_weights_destination_path, "exported_weights"
        )

        # Create directories if they don't exist
        for directory in [
            self.weights_dir,
            self.pretrained_weights_dir,
            self.trained_weights_dir,
            self.config_dir,
            self.exported_weights_dir,
        ]:
            os.makedirs(directory, exist_ok=True)

        # Download and process model files
        for model_file in self.model_version.list_files():
            if model_file.name == self.pretrained_weights_name:
                self.pretrained_weights_path = downloader.download_and_process(
                    model_file, self.pretrained_weights_dir
                )
            elif model_file.name == self.trained_weights_name:
                self.trained_weights_path = downloader.download_and_process(
                    model_file, self.trained_weights_dir
                )
            elif model_file.name == self.config_name:
                self.config_path = downloader.download_and_process(
                    model_file, self.config_dir
                )
            elif model_file.name == self.exported_weights_name:
                self.exported_weights_path = downloader.download_and_process(
                    model_file, self.exported_weights_dir
                )
            else:
                downloader.download_and_process(
                    model_file, model_weights_destination_path
                )


TModelContext = TypeVar("TModelContext", bound=ModelContext)
