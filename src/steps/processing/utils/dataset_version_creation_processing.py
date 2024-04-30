from abc import abstractmethod
from typing import List, Optional, Tuple

from picsellia import Client, DatasetVersion, Data
from picsellia.services.error_manager import ErrorManager
from picsellia.types.enums import InferenceType


class DatasetVersionCreationProcessing:
    """
    Handles the processing of creating a dataset version.

    This class offers all the necessary methods to handle a processing of type DatasetVersionCreation of Picsellia.
    It allows to upload images to the datalake, add them to a dataset version, and update the dataset version with
    the necessary information.

    Attributes:
        client (Client): The Picsellia client to use for the processing.
        output_dataset_version (DatasetVersion): The dataset version to create.
        dataset_type (InferenceType): The type of dataset to create.
        dataset_description (str): The description of the dataset to create.
        datalake_name (str): The name of the datalake to use for the processing.
    """

    def __init__(
        self,
        client: Client,
        output_dataset_version: DatasetVersion,
        dataset_type: InferenceType,
        dataset_description: str,
        datalake: str = "default",
    ):
        self.client = client
        self.output_dataset_version = output_dataset_version
        self.output_dataset_version.update(
            description=dataset_description, type=dataset_type
        )
        self.datalake = self.client.get_datalake(name=datalake)

    def _upload_data_with_error_manager(
        self, images_to_upload: List[str], images_tags: Optional[List[str]] = None
    ) -> Tuple[Data, List[str]]:
        """
        Uploads data to the datalake using an error manager. This method allows to handle errors during the upload process.
        It will retry to upload the data that failed to upload.

        Args:
            images_to_upload (List[str]): The list of image file paths to upload.
            images_tags (Optional[List[str]]): The list of tags to associate with the images.

        Returns:

        """
        error_manager = ErrorManager()
        data = self.datalake.upload_data(
            filepaths=images_to_upload, tags=images_tags, error_manager=error_manager
        )
        error_paths = [error.path for error in error_manager.errors]
        return data, error_paths

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
        all_uploaded_data.extend(
            [one_uploaded_data for one_uploaded_data in uploaded_data]
        )
        retry_count = 0
        while error_paths and retry_count < max_retries:
            uploaded_data, error_paths = self._upload_data_with_error_manager(
                images_to_upload=error_paths, images_tags=images_tags
            )
            all_uploaded_data.extend(
                [one_uploaded_data for one_uploaded_data in uploaded_data]
            )
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
        adding_job = self.output_dataset_version.add_data(data=data)
        adding_job.wait_for_done()

    @abstractmethod
    def process(self) -> None:
        """
        Processes the dataset version creation.
        """
        pass
