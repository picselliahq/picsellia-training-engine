import os
import shutil
from typing import Dict
from src.steps.data_extraction.utils.dataset_context import DatasetContext


class ClassificationDatasetOrganizer:
    def __init__(self, dataset_context: DatasetContext):
        self.dataset_context = dataset_context
        self.destination_dir = dataset_context.dataset_extraction_path

    def organize(self):
        categories = self._extract_categories()
        image_categories = self._map_image_to_category()
        self._organize_images(categories, image_categories)
        self._cleanup()
        return self.dataset_context

    def _extract_categories(self) -> Dict[int, str]:
        return {
            category.id: category.name
            for category in self.dataset_context.coco_file.categories
        }

    def _map_image_to_category(self) -> Dict[str, int]:
        return {
            annotation.image_id: annotation.category_id
            for annotation in self.dataset_context.coco_file.annotations
        }

    def _organize_images(
        self, categories: Dict[int, str], image_categories: Dict[str, int]
    ):
        for image in self.dataset_context.coco_file.images:
            image_id = image.id
            if image_id in image_categories:
                category_id = image_categories[image_id]
                category_name = categories[category_id]
                self._create_category_dir_and_copy_image(category_name, image)

    def _create_category_dir_and_copy_image(self, category_name: str, image):
        category_dir = os.path.join(self.destination_dir, category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        src_image_path = os.path.join(self.dataset_context.image_dir, image.file_name)
        dest_image_path = os.path.join(category_dir, image.file_name)
        shutil.copy(src_image_path, dest_image_path)

    def _cleanup(self):
        shutil.rmtree(self.dataset_context.image_dir)
        self.dataset_context.image_dir = self.destination_dir
