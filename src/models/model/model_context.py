import os
import tarfile
import zipfile
from abc import abstractmethod
from typing import Optional, Dict

from picsellia import ModelVersion, Label


class ModelContext:
    """
    This class is used to store the context of an AI model, which includes its metadata, paths to its assets and weights.

    Attributes:
        - model_name: The name of the model.
        - model_version: The version of the model, as managed by Picsellia.
        - multi_asset: A collection of assets associated with the model.
        - labelmap: A mapping from label names to (Picsellia) label objects.
        - destination_path: The root path where the model should be stored locally.
        - model_weights_path: The path where the model weights are stored.
        - config_file_path: The configuration file associated with the model.
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
            labelmap (dict): The mapping of label names to ids.
        """
        self.model_name = model_name
        self.prefix_model_name = prefix_model_name
        self.model_version = model_version
        self.destination_path = destination_path
        self.labelmap = labelmap
        self.pretrained_model_path = None
        self.config_file_path = None
        self.loaded_model = None
        self.model_weights_path = self.get_model_weights_path()
        self.trained_model_path = self.get_trained_model_path()
        self.results_path = self.get_results_path()
        self.inference_model_path = self.get_inference_model_path()
        os.makedirs(self.model_weights_path, exist_ok=True)
        os.makedirs(self.trained_model_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.makedirs(self.inference_model_path, exist_ok=True)

    @abstractmethod
    def load_model(self):
        """
        Loads the model from the model weights.
        """
        pass

    def download_weights(self) -> None:
        """
        Downloads the model weights to a local directory.
        """
        os.makedirs(self.model_weights_path, exist_ok=True)
        model_files = self.model_version.list_files()
        for model_file in model_files:
            print(f"Downloading model file: {model_file.name}")
            self.download_model_file(model_file)

    def download_model_file(self, model_file):
        if self.prefix_model_name:
            if model_file.name.startswith(f"{self.prefix_model_name}-"):
                model_file.download(self.model_weights_path)
                self.extract_weights(model_file=model_file)
            if model_file.name == f"{self.prefix_model_name}-config":
                self.config_file_path = self.get_extracted_path(model_file)
            elif model_file.name == f"{self.prefix_model_name}-pretrained-model":
                self.pretrained_model_path = self.get_extracted_path(model_file)
            elif (
                model_file.name == f"{self.prefix_model_name}-model-latest"
                and not self.pretrained_model_path
            ):
                self.pretrained_model_path = self.get_extracted_path(model_file)
            elif (
                model_file.name == f"{self.prefix_model_name}-weights"
                and not self.pretrained_model_path
            ):
                self.pretrained_model_path = self.get_extracted_path(model_file)
        else:
            model_file.download(self.model_weights_path)
            print(f"Downloaded model file: {model_file.name}")
            self.extract_weights(model_file=model_file)
            if model_file.name == "config":
                self.config_file_path = self.get_extracted_path(model_file)
            elif model_file.name == "pretrained-model":
                self.pretrained_model_path = self.get_extracted_path(model_file)
            elif model_file.name == "model-latest" and not self.pretrained_model_path:
                self.pretrained_model_path = self.get_extracted_path(model_file)
            elif model_file.name == "weights" and not self.pretrained_model_path:
                self.pretrained_model_path = self.get_extracted_path(model_file)

    def get_extracted_path(self, model_file):
        if model_file.filename.endswith(".tar") or model_file.filename.endswith(".zip"):
            return os.path.join(self.model_weights_path, model_file.filename[:-4])
        else:
            return os.path.join(self.model_weights_path, model_file.filename)

    def extract_weights(self, model_file):
        if model_file.filename.endswith(".tar"):
            with tarfile.open(
                os.path.join(self.model_weights_path, model_file.filename), "r:*"
            ) as tar:
                tar.extractall(path=self.model_weights_path)
        elif model_file.filename.endswith(".zip"):
            with zipfile.ZipFile(
                os.path.join(self.model_weights_path, model_file.filename), "r"
            ) as zipf:
                zipf.extractall(path=self.model_weights_path)

        if model_file.filename.endswith((".tar", ".zip")):
            os.remove(os.path.join(self.model_weights_path, model_file.filename))

    def get_model_weights_path(self):
        if self.prefix_model_name:
            return os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                "weights",
            )
        else:
            return os.path.join(self.destination_path, self.model_name, "weights")

    def get_trained_model_path(self):
        if self.prefix_model_name:
            return os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                "trained_model",
            )
        else:
            return os.path.join(self.destination_path, self.model_name, "trained_model")

    def get_results_path(self):
        if self.prefix_model_name:
            return os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                "results",
            )
        else:
            return os.path.join(self.destination_path, self.model_name, "results")

    def get_inference_model_path(self):
        if self.prefix_model_name:
            return os.path.join(
                self.destination_path,
                self.model_name,
                self.prefix_model_name,
                "inference_model",
            )
        else:
            return os.path.join(
                self.destination_path, self.model_name, "inference_model"
            )

    def get_config_file_path(self):
        if self.prefix_model_name:
            return os.path.join(
                self.model_weights_path, f"{self.prefix_model_name}-config"
            )
        else:
            return os.path.join(self.model_weights_path, "config")
