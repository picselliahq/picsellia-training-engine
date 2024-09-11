import os
import tarfile
import zipfile

from picsellia import ModelFile


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
