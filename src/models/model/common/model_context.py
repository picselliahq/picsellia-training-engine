import os
import tarfile
import zipfile
from abc import abstractmethod
from typing import Optional, Dict, Any

from picsellia import ModelVersion, Label, ModelFile


class ModelContext:
    """
    Stores the context of an AI model, including metadata and file paths.

    Attributes:
        model_name (str): The name of the model.
        model_version (ModelVersion): The version of the model, managed by Picsellia.
        destination_path (str): The root path where the model is stored locally.
        labelmap (Optional[Dict[str, Label]]): A mapping from label names to Picsellia label objects.
        prefix_model_name (Optional[str]): Optional prefix used in model file names.
        pretrained_model_path (Optional[str]): Path to the pretrained model file.
        config_file_path (Optional[str]): Path to the configuration file.
        _loaded_model (Optional[Any]): Loaded model instance, initially None.
        model_weights_path (str): Path where model weights are stored.
        trained_model_path (str): Path where the trained model is stored.
        results_path (str): Path where results are stored.
        inference_model_path (str): Path where the inference model is stored.
    """

    def __init__(
        self,
        model_name: str,
        model_version: ModelVersion,
        destination_path: str,
        labelmap: Optional[Dict[str, Label]] = None,
        prefix_model_name: Optional[str] = None,
    ):
        """
        Initializes the ModelContext with model metadata and configuration.

        Args:
            model_name (str): The name of the model.
            model_version (ModelVersion): The model version object.
            destination_path (str): The root directory for storing the model locally.
            labelmap (Optional[Dict[str, Label]]): The mapping of label names to label objects.
            prefix_model_name (Optional[str]): Optional prefix used in model file names.
        """
        self.model_name = model_name
        self.model_version = model_version
        self.destination_path = destination_path
        self.labelmap = labelmap or {}
        self.prefix_model_name = prefix_model_name

        self.pretrained_model_path: Optional[str] = None
        self.config_file_path: Optional[str] = None
        self._loaded_model: Optional[Any] = None

        self.model_weights_path = self._get_model_weights_path()
        self.trained_model_path = self._get_trained_model_path()
        self.results_path = self._get_results_path()
        self.inference_model_path = self._get_inference_model_path()
        self._create_directories()

    @property
    def loaded_model(self) -> Any:
        """
        Returns the loaded model instance. Raises an error if the model is not loaded.

        Returns:
            Any: The loaded model instance.

        Raises:
            ValueError: If the model is not loaded.
        """
        if self._loaded_model is None:
            raise ValueError("Model is not loaded")
        return self._loaded_model

    @loaded_model.setter
    def loaded_model(self, value: Any):
        """
        Sets the loaded model instance.

        Args:
            value (Any): The model instance to set as loaded.
        """
        self._loaded_model = value

    @abstractmethod
    def load_model(self) -> None:
        """Abstract method to be implemented by subclasses for loading the model."""
        pass

    def download_weights(self) -> None:
        """
        Downloads model weights and configuration files to the local directory.
        """
        os.makedirs(self.model_weights_path, exist_ok=True)
        model_files = self.model_version.list_files()
        for model_file in model_files:
            print(f"Downloading model file: {model_file.name}")
            self._download_and_process_model_file(model_file)

    def _download_and_process_model_file(self, model_file: ModelFile) -> None:
        """
        Downloads a model file and processes it by extracting if needed and setting paths.

        Args:
            model_file (ModelFile): The model file to download and process.
        """
        model_file.download(self.model_weights_path)
        self._unzip_if_needed(model_file)

        if self._is_config_file(model_file.name):
            self.config_file_path = self._get_extracted_path(model_file)
        elif self._is_pretrained_model_file(model_file.name):
            self._set_pretrained_model_path_if_none(model_file)

    def _get_extracted_path(self, model_file: ModelFile) -> str:
        """
        Determines the path where the model file will be extracted.

        Args:
            model_file (ModelFile): The model file to extract.

        Returns:
            str: The path to the extracted file.
        """
        if model_file.filename.endswith((".tar", ".zip")):
            return os.path.join(self.model_weights_path, model_file.filename[:-4])
        return os.path.join(self.model_weights_path, model_file.filename)

    def _unzip_if_needed(self, model_file: ModelFile) -> None:
        """
        Unzips or extracts the model file if it's a .tar or .zip file.

        Args:
            model_file (ModelFile): The model file to unzip or extract.
        """
        file_path = os.path.join(self.model_weights_path, model_file.filename)
        if model_file.filename.endswith(".tar"):
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=self.model_weights_path)
        elif model_file.filename.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zipf:
                zipf.extractall(path=self.model_weights_path)
        if model_file.filename.endswith((".tar", ".zip")):
            os.remove(file_path)

    def _set_pretrained_model_path_if_none(self, model_file: ModelFile):
        """
        Sets the pretrained model path based on the file name hierarchy.

        Args:
            model_file (ModelFile): The model file to set as pretrained.
        """
        if model_file.name == "pretrained-model":
            self.pretrained_model_path = self._get_extracted_path(model_file)
        elif model_file.name == "model-latest" and not self.pretrained_model_path:
            self.pretrained_model_path = self._get_extracted_path(model_file)
        elif model_file.name == "weights" and not self.pretrained_model_path:
            self.pretrained_model_path = self._get_extracted_path(model_file)

    def _get_model_weights_path(self) -> str:
        """
        Constructs the path for storing model weights.

        Returns:
            str: The path to the model weights directory.
        """
        return self._construct_path("weights")

    def _get_trained_model_path(self) -> str:
        """
        Constructs the path for storing the trained model.

        Returns:
            str: The path to the trained model directory.
        """
        return self._construct_path("trained_model")

    def _get_results_path(self) -> str:
        """
        Constructs the path for storing model results.

        Returns:
            str: The path to the results directory.
        """
        return self._construct_path("results")

    def _get_inference_model_path(self) -> str:
        """
        Constructs the path for storing the inference model.

        Returns:

        """
        return self._construct_path("inference_model")

    def _construct_path(self, folder_name: str) -> str:
        """
        Constructs the path to a folder within the model directory.

        Args:
            folder_name: The name of the folder to construct the path for.

        Returns:

        """
        if self.prefix_model_name:
            return os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                folder_name,
            )
        return os.path.join(self.destination_path, self.model_name, folder_name)

    def _create_directories(self) -> None:
        """
        Creates the directories for storing model files.

        Returns:

        """
        os.makedirs(self.model_weights_path, exist_ok=True)
        os.makedirs(self.trained_model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.inference_model_path, exist_ok=True)

    def _is_config_file(self, file_name: str) -> bool:
        """
        Determines if a file is a configuration file based on the file name.

        Args:
            file_name: The name of the file to check.

        Returns:

        """
        return (
            file_name == f"{self.prefix_model_name}-config"
            if self.prefix_model_name
            else file_name == "config"
        )

    def _is_pretrained_model_file(self, file_name: str) -> bool:
        """
        Determines if a file is a pretrained model file based on the file name.

        Args:
            file_name: The name of the file to check.

        Returns:

        """
        if self.prefix_model_name:
            return file_name in {
                f"{self.prefix_model_name}-pretrained-model",
                f"{self.prefix_model_name}-model-latest",
                f"{self.prefix_model_name}-weights",
            }
        return file_name in {"pretrained-model", "model-latest", "weights"}

    def get_config_file_path(self) -> str:
        """
        Gets the path to the configuration

        Returns:

        """
        if self.prefix_model_name:
            return os.path.join(
                self.model_weights_path, f"{self.prefix_model_name}-config"
            )
        return os.path.join(self.model_weights_path, "config")
