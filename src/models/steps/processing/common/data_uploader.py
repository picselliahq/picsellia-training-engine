from typing import List, Optional, Tuple

from picsellia import Client, DatasetVersion, Data
from picsellia.services.error_manager import ErrorManager


class DataUploader:
    """
    Handles the process of creating a dataset version in Picsellia.

    This class provides methods to upload images to a datalake, add them to a dataset version, and
    update the dataset version with the necessary information, including COCO annotations.

    Attributes:
        client (Client): The Picsellia client to interact with the API.
        dataset_version (DatasetVersion): The dataset version that will be updated with images and annotations.
        datalake (str): The datalake to which images will be uploaded (default is 'default').
    """

    def __init__(
        self, client: Client, dataset_version: DatasetVersion, datalake: str = "default"
    ):
        """
        Initializes the DataUploader with a Picsellia client, dataset version, and optional datalake.

        Args:
            client (Client): The Picsellia client used for communication with the API.
            dataset_version (DatasetVersion): The dataset version that will be updated.
            datalake (str): The name of the datalake where images will be uploaded (default is 'default').
        """
        self.client = client
        self.dataset_version = dataset_version
        self.datalake = self.client.get_datalake(name=datalake)

    def _upload_data_with_error_manager(
        self, images_to_upload: List[str], data_tags: Optional[List[str]] = None
    ) -> Tuple[List[Data], List[str]]:
        """
        Uploads images to the datalake using an error manager to handle failed uploads.

        This method attempts to upload images to the datalake. If any uploads fail, they are retried
        using the error manager. It returns the successfully uploaded data and a list of paths
        that failed to upload.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            data_tags (Optional[List[str]]): The list of tags to associate with the uploaded images (default is None).

        Returns:
            Tuple[List[Data], List[str]]: A tuple containing:
                - List[Data]: A list of successfully uploaded data.
                - List[str]: A list of file paths that failed to upload.
        """
        error_manager = ErrorManager()
        data = self.datalake.upload_data(
            filepaths=images_to_upload, tags=data_tags, error_manager=error_manager
        )

        if isinstance(data, Data):
            uploaded_data = [data]
        else:
            uploaded_data = [one_uploaded_data for one_uploaded_data in data]

        error_paths = [error.path for error in error_manager.errors]
        return uploaded_data, error_paths

    def _upload_images_to_datalake(
        self,
        images_to_upload: List[str],
        data_tags: Optional[List[str]] = None,
        max_retries: int = 5,
    ) -> List[Data]:
        """
        Uploads images to the datalake and retries failed uploads up to a maximum number of retries.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            data_tags (Optional[List[str]]): The list of tags to associate with the images (default is None).
            max_retries (int): The maximum number of retry attempts for failed uploads (default is 5).

        Returns:
            List[Data]: A list of successfully uploaded data.

        Raises:
            Exception: If the maximum number of retries is exceeded and there are still failed uploads.
        """
        all_uploaded_data = []
        uploaded_data, error_paths = self._upload_data_with_error_manager(
            images_to_upload=images_to_upload, data_tags=data_tags
        )
        all_uploaded_data.extend(uploaded_data)
        retry_count = 0
        while error_paths and retry_count < max_retries:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                images_to_upload=error_paths, data_tags=data_tags
            )
            all_uploaded_data.extend(uploaded_data)
            retry_count += 1
        if error_paths:
            raise Exception(
                f"Failed to upload the following images: {error_paths} after {max_retries} retries."
            )
        return all_uploaded_data

    def _add_images_to_dataset_version(
        self,
        images_to_upload: List[str],
        data_tags: Optional[List[str]] = None,
        asset_tags: Optional[List[str]] = None,
        max_retries: int = 5,
        attempts: int = 150,
        blocking_time_increment: float = 50.0,
    ) -> None:
        """
        Adds uploaded images to the dataset version and waits for the process to complete.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            data_tags (Optional[List[str]]): The list of tags to associate with the images (default is None).
            asset_tags (Optional[List[str]]): The list of tags to associate with the dataset version assets (default is None).
            max_retries (int): The maximum number of retry attempts for failed uploads (default is 5).
            attempts (int): The number of attempts to wait for the adding job to complete (default is 150).
            blocking_time_increment (float): The amount of time to wait between job status checks (default is 50.0 seconds).
        """
        data = self._upload_images_to_datalake(
            images_to_upload=images_to_upload,
            data_tags=data_tags,
            max_retries=max_retries,
        )
        adding_job = self.dataset_version.add_data(data=data, tags=asset_tags)
        adding_job.wait_for_done(
            blocking_time_increment=blocking_time_increment, attempts=attempts
        )

    def _add_coco_annotations_to_dataset_version(self, annotation_path: str):
        """
        Adds COCO annotations to the dataset version.

        Args:
            annotation_path (str): The path to the COCO annotations file.
        """
        self.dataset_version.import_annotations_coco_file(file_path=annotation_path)
