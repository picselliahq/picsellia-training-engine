import os
import tarfile
import zipfile
from typing import Optional, Dict, Any

from picsellia import ModelFile
from picsellia import ModelVersion, Label


class ModelDownloader:
    def download_and_process(self, model_file: ModelFile, destination_path: str) -> str:
        """
        Télécharge et décompresse un fichier de modèle.

        Args:
            model_file (ModelFile): Le fichier de modèle à télécharger.

        Returns:
            str: Le chemin vers le fichier téléchargé ou décompressé.
        """
        # Assure que le répertoire de destination existe
        os.makedirs(destination_path, exist_ok=True)

        # Chemin complet du fichier à télécharger
        file_path = os.path.join(destination_path, model_file.filename)

        # Téléchargement du fichier
        model_file.download(destination_path)

        # Décompression si nécessaire et retour du chemin du fichier final
        return self._unzip_if_needed(
            file_path=file_path, destination_path=destination_path
        )

    def _unzip_if_needed(self, file_path: str, destination_path: str) -> str:
        """
        Décompresse un fichier si nécessaire et retourne le chemin du fichier extrait.

        Args:
            file_path (str): Le chemin du fichier à décompresser.

        Returns:
            str: Le chemin du fichier extrait ou du fichier d'origine s'il n'était pas compressé.
        """
        if file_path.endswith(".tar"):
            with tarfile.open(file_path, "r:*") as tar:
                tar.extractall(path=destination_path)
            os.remove(file_path)
            return file_path[:-4]

        elif file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zipf:
                zipf.extractall(path=destination_path)
            os.remove(file_path)
            return file_path[:-4]

        return file_path


class ModelContext:
    def __init__(
        self,
        model_name: str,
        model_version: ModelVersion,
        destination_path: str,
        pretrained_model_filename: str = "model-latest",
        config_filename: str = "config",
        labelmap: Optional[Dict[str, Label]] = None,
        prefix_model_name: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.destination_path = destination_path
        self.pretrained_model_filename = pretrained_model_filename
        self.config_filename = config_filename
        self.labelmap = labelmap or {}
        self.prefix_model_name = prefix_model_name

        self.weights_dir = self._construct_path("weights")
        self.results_dir = self._construct_path("results")
        self.inference_model_dir = self._construct_path("inference_model")

        self.pretrained_model_path: Optional[str] = None
        self.config_file_path: Optional[str] = None
        self._loaded_model: Optional[Any] = None

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
            raise ValueError(
                "Model is not loaded. Please load the model before accessing it."
            )
        return self._loaded_model

    def set_loaded_model(self, model: Any) -> None:
        """
        Sets the loaded model instance.

        Args:
            model (Any): The model instance to set as loaded.
        """
        self._loaded_model = model

    def download_weights(self, model_weights_destination_path: str) -> None:
        """
        Télécharge les fichiers de poids et de configuration, en les assignant aux bons chemins.
        """
        downloader = ModelDownloader()

        # Boucle sur tous les fichiers du modèle
        model_files = self.model_version.list_files()
        for model_file in model_files:
            file_path = downloader.download_and_process(
                model_file=model_file, destination_path=model_weights_destination_path
            )

            # Assigne les chemins selon le nom du fichier
            if model_file.filename == self.pretrained_model_filename:
                self.pretrained_model_path = file_path
            elif model_file.filename == self.config_filename:
                self.config_file_path = file_path

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
