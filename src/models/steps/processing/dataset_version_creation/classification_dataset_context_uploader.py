import logging
import os
from typing import List, Optional

from picsellia import Client
from picsellia.types.enums import TagTarget

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.data_uploader import (
    DataUploader,
)

logger = logging.getLogger("picsellia")


class ClassificationDatasetContextUploader(DataUploader):
    def __init__(
        self,
        client: Client,
        dataset_context: DatasetContext,
        datalake: str = "default",
        data_tags: Optional[List[str]] = None,
    ):
        super().__init__(client, dataset_context.dataset_version, datalake)
        self.client = client
        self.dataset_context = dataset_context
        self.datalake = self.client.get_datalake(name=datalake)
        self.data_tags = data_tags

    def upload_dataset_context(self) -> None:
        """
        Uploads the dataset context to Picsellia.
        """
        for label_folder in os.listdir(self.dataset_context.image_dir):
            full_label_folder_path = os.path.join(
                self.dataset_context.image_dir, label_folder
            )
            if os.path.isdir(full_label_folder_path):
                filepaths = [
                    os.path.join(full_label_folder_path, file)
                    for file in os.listdir(full_label_folder_path)
                ]
                self._add_images_to_dataset_version(
                    images_to_upload=filepaths,
                    data_tags=self.data_tags,
                    asset_tags=[label_folder],
                )

        conversion_job = (
            self.dataset_context.dataset_version.convert_tags_to_classification(
                tag_type=TagTarget.ASSET,
                tags=self.dataset_context.dataset_version.list_asset_tags(),
            )
        )
        conversion_job.wait_for_done()
