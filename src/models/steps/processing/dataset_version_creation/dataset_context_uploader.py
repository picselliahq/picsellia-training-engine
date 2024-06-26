import os
from typing import List, Optional

from picsellia import Client

from src.models.steps.processing.dataset_version_creation.data_uploader import (
    DataUploader,
)
from src.models.dataset.common.dataset_context import DatasetContext


class DatasetContextUploader(DataUploader):
    def __init__(
        self,
        client: Client,
        dataset_context: DatasetContext,
        datalake: str = "default",
        images_tags: Optional[List[str]] = None,
    ):
        super().__init__(client, dataset_context.dataset_version, datalake)
        self.client = client
        self.dataset_context = dataset_context
        self.datalake = self.client.get_datalake(name=datalake)
        self.images_tags = images_tags

    def upload_dataset_context(self) -> None:
        """
        Uploads the dataset context to Picsellia.
        """
        self._add_images_to_dataset_version(
            images_to_upload=[
                os.path.join(self.dataset_context.image_dir, image_filename)
                for image_filename in os.listdir(self.dataset_context.image_dir)
            ],
            images_tags=self.images_tags,
        )
        self._add_coco_annotations_to_dataset_version(
            annotation_path=self.dataset_context.coco_file_path
        )
