import logging
import os
from typing import List, Optional

from picsellia import Client
from picsellia.types.enums import InferenceType

from src.models.steps.processing.common.data_uploader import DataUploader
from src.models.dataset.common.dataset_context import DatasetContext

logger = logging.getLogger("picsellia")


class SegmentationDatasetContextUploader(DataUploader):
    """
    Handles uploading the dataset context for segmentation tasks to Picsellia.

    This class extends `DataUploader` and specifically focuses on segmentation datasets.
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
    ):
        """
        Initializes the SegmentationDatasetContextUploader with a client, dataset context, and optional datalake.

        Args:
            client (Client): The Picsellia client used for API interactions.
            dataset_context (DatasetContext): The dataset context containing images and annotations.
            datalake (str): The name of the datalake where the images will be uploaded (default is 'default').
            data_tags (Optional[List[str]]): Optional tags to associate with the uploaded images.
        """
        super().__init__(client, dataset_context.dataset_version, datalake)
        self.client = client
        self.dataset_context = dataset_context
        self.datalake = self.client.get_datalake(name=datalake)
        self.data_tags = data_tags

    def upload_dataset_context(self) -> None:
        """
        Uploads the dataset context to Picsellia, including images and annotations.

        This method uploads the images from the dataset context to the datalake. If the dataset version
        is configured for segmentation tasks (i.e., its type is not `NOT_CONFIGURED`), it will also upload
        COCO annotations. Otherwise, it logs that no annotations will be uploaded due to the dataset type.
        """
        # Upload images to the datalake
        if self.dataset_context.images_dir:
            self._add_images_to_dataset_version(
                images_to_upload=[
                    os.path.join(self.dataset_context.images_dir, image_filename)
                    for image_filename in os.listdir(self.dataset_context.images_dir)
                ],
                data_tags=self.data_tags,
            )

        # Check the dataset type and conditionally upload annotations
        if (
            self.dataset_context.dataset_version.type != InferenceType.NOT_CONFIGURED
            and self.dataset_context.coco_file_path
        ):
            self._add_coco_annotations_to_dataset_version(
                annotation_path=self.dataset_context.coco_file_path
            )
        else:
            logger.info(
                f"👉 Since the dataset's type is set to {InferenceType.NOT_CONFIGURED.name}, "
                f"no annotations will be uploaded."
            )
