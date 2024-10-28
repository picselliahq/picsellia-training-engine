import logging
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

from picsellia import Client
from picsellia.types.enums import TagTarget

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.common.data_uploader import DataUploader

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
        Uploads the dataset context to Picsellia using COCO annotation file.
        """
        coco_data = self.dataset_context.load_coco_file_data()
        images_by_category = self._process_coco_data(coco_data)

        for category_name, image_paths in images_by_category.items():
            existing_paths = [path for path in image_paths if os.path.exists(path)]
            if existing_paths:
                self._add_images_to_dataset_version(
                    images_to_upload=existing_paths,
                    data_tags=self.data_tags,
                    asset_tags=[category_name],
                )

            missing_paths = set(image_paths) - set(existing_paths)
            if missing_paths:
                logger.warning(
                    f"The following image files were not found for category '{category_name}': {missing_paths}"
                )

        conversion_job = (
            self.dataset_context.dataset_version.convert_tags_to_classification(
                tag_type=TagTarget.ASSET,
                tags=self.dataset_context.dataset_version.list_asset_tags(),
            )
        )
        conversion_job.wait_for_done()

    def _process_coco_data(self, coco_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Process COCO data to group image paths by category name.
        """
        if not self.dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        category_id_to_name = {
            cat["id"]: cat["name"] for cat in coco_data["categories"]
        }
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}

        images_by_category = defaultdict(list)
        for annotation in coco_data["annotations"]:
            image_info = image_id_to_info[annotation["image_id"]]
            category_name = category_id_to_name[annotation["category_id"]]
            image_path = os.path.join(
                self.dataset_context.images_dir, image_info["file_name"]
            )

            images_by_category[category_name].append(image_path)

        return dict(images_by_category)
