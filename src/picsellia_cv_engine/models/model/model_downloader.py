import os
import tarfile
import zipfile

from picsellia import ModelFile


class ModelDownloader:
    def download_and_process(self, model_file: ModelFile, destination_path: str) -> str:
        """
        Downloads and extracts a model file if necessary.

        Args:
            model_file (ModelFile): The model file to download.
            destination_path (str): The destination path where the file will be stored.

        Returns:
            str: The path to the downloaded or extracted file.
        """
        os.makedirs(destination_path, exist_ok=True)
        file_path = os.path.join(destination_path, model_file.filename)
        model_file.download(destination_path)

        return self._unzip_if_needed(
            file_path=file_path, destination_path=destination_path
        )

    def _unzip_if_needed(self, file_path: str, destination_path: str) -> str:
        """
        Extracts a compressed file if it's in a .tar or .zip format.

        Args:
            file_path (str): The path of the file to extract.
            destination_path (str): The directory where the file should be extracted.

        Returns:
            str: The path to the extracted file, or the original file if no extraction was needed.
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
