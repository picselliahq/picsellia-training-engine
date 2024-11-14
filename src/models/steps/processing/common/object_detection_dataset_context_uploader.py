import logging
import os
from typing import List, Optional

from picsellia import Client
from picsellia.types.enums import InferenceType

from src.models.steps.processing.common.data_uploader import DataUploader
from src.models.dataset.common.dataset_context import DatasetContext

logger = logging.getLogger("picsellia")


class ObjectDetectionDatasetContextUploader(DataUploader):
    """
    Handles uploading the dataset context for object detection tasks to Picsellia.

    This class extends `DataUploader` and specifically focuses on object detection datasets.
    It uploads images to a specified datalake and, if the dataset version is correctly configured,
    uploads COCO annotations as well.

    Attributes:
        client (Client): The Picsellia client used for API interactions.
        dataset_context (DatasetContext): The context containing the dataset's images and annotations.
        datalake (str): The datalake to which the images will be uploaded.
        data_tags (Optional[List[str]]): Optional tags to associate with the uploaded data.
    """

    def __init__(
        self,
        client: Client,
        dataset_context: DatasetContext,
        datalake: str = "default",
        data_tags: Optional[List[str]] = None,
        batch_size: int = 10000,
        use_id: bool = True,
        fail_on_asset_not_found: bool = True,
    ):
        """
        Initializes the ObjectDetectionDatasetContextUploader with a client, dataset context, and optional datalake.

        Args:
            client (Client): The Picsellia client used for API interactions.
            dataset_context (DatasetContext): The dataset context containing images and annotations.
            datalake (str): The name of the datalake where the images will be uploaded (default is 'default').
            data_tags (Optional[List[str]]): Optional tags to associate with the uploaded images.
            batch_size (int): The number of images per batch for uploading (default is 10).
        """
        super().__init__(client, dataset_context.dataset_version, datalake)
        self.client = client
        self.dataset_context = dataset_context
        self.datalake = self.client.get_datalake(name=datalake)
        self.data_tags = data_tags
        self.batch_size = batch_size
        self.use_id = use_id
        self.fail_on_asset_not_found = fail_on_asset_not_found

    def upload_dataset_context(self) -> None:
        """
        Uploads the dataset context to Picsellia, including images and annotations.

        This method uploads the images from the dataset context to the datalake in batches. If the dataset version
        is configured for object detection (i.e., its type is not `NOT_CONFIGURED`), it will also upload
        COCO annotations in batches. Otherwise, it logs that no annotations will be uploaded due to the dataset type.
        """
        # Upload images to the datalake in batches
        if self.dataset_context.images_dir:
            self._add_images_to_dataset_version_in_batches(
                images_to_upload=[
                    os.path.join(self.dataset_context.images_dir, image_filename)
                    for image_filename in os.listdir(self.dataset_context.images_dir)
                ],
                data_tags=self.data_tags,
                batch_size=self.batch_size,
            )

        # Check the dataset type and conditionally upload annotations in batches
        if (
            self.dataset_context.dataset_version.type != InferenceType.NOT_CONFIGURED
            and self.dataset_context.coco_file_path
        ):
            print("Uploading annotations")
            self._add_coco_annotations_to_dataset_version_in_batches(
                annotation_path=self.dataset_context.coco_file_path,
                batch_size=self.batch_size,
                use_id=self.use_id,
                fail_on_asset_not_found=self.fail_on_asset_not_found,
            )
        else:
            logger.info(
                f"👉 Since the dataset's type is set to {InferenceType.NOT_CONFIGURED.name}, "
                f"no annotations will be uploaded."
            )
