from typing import List, Optional, Tuple

from picsellia import Client, DatasetVersion, Data
from picsellia.services.error_manager import ErrorManager


class DataUploader:
    """
    Handles the processing of creating a dataset version.

    This class offers all the necessary methods to handle a processing of type DatasetVersionCreation of Picsellia.
    It allows to upload images to the datalake, add them to a dataset version, and update the dataset version with
    the necessary information.
    """

    def __init__(
        self,
        client: Client,
        dataset_version: DatasetVersion,
        datalake: str = "default",
    ):
        self.client = client
        self.dataset_version = dataset_version
        self.datalake = self.client.get_datalake(name=datalake)

    def _upload_data_with_error_manager(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> Tuple[List[Data], List[str]]:
        """
        Uploads data to the datalake using an error manager. This method allows to handle errors during the upload process.
        It will retry to upload the data that failed to upload.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            images_tags (Optional[List[str]]): The list of tags to associate with the images.

        Returns:
            - List[Data]: The list of uploaded data.
            - List[str]: The list of file paths that failed to upload.
        """
        error_manager = ErrorManager()
        data = self.datalake.upload_data(
            filepaths=images_to_upload, tags=images_tags, error_manager=error_manager
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
        images_tags: Optional[List[str]] = None,
        max_retries: int = 5,
    ) -> List[Data]:
        """
        Uploads images to the datalake. This method allows to handle errors during the upload process.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            images_tags (Optional[List[str]]): The list of tags to associate with the images.
            max_retries (int): The maximum number of retries to upload the images.

        Returns:

        """
        all_uploaded_data = []
        uploaded_data, error_paths = self._upload_data_with_error_manager(
            images_to_upload=images_to_upload, images_tags=images_tags
        )
        all_uploaded_data.extend(uploaded_data)
        retry_count = 0
        while error_paths and retry_count < max_retries:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                images_to_upload=error_paths, images_tags=images_tags
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
        images_tags: Optional[List[str]] = None,
        max_retries: int = 5,
        attempts: int = 150,
    ) -> None:
        """
        Adds images to the dataset version.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            images_tags (Optional[List[str]]): The list of tags to associate with the images.
            max_retries (int): The maximum number of retries to upload the images.

        """
        data = self._upload_images_to_datalake(
            images_to_upload=images_to_upload,
            images_tags=images_tags,
            max_retries=max_retries,
        )
        adding_job = self.dataset_version.add_data(data=data)
        adding_job.wait_for_done(attempts=attempts)

    def _add_coco_annotations_to_dataset_version(self, annotation_path: str):
        """
        Adds COCO annotations to the dataset version.

        Args:
            annotation_path (str): The path to the COCO annotations file.

        """
        self.dataset_version.import_annotations_coco_file(file_path=annotation_path)
