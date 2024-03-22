import os
import shutil

from src.steps.data_extraction.utils.dataset_context import DatasetContext


class ClassificationDatasetOrganizer:
    def __init__(self, dataset_context: DatasetContext):
        self.dataset_context = dataset_context
        self.destination_dir = dataset_context.dataset_extraction_path

    def organize(self):
        categories = {
            category.id: category.name
            for category in self.dataset_context.coco_file.categories
        }
        image_categories = {}
        for annotation in self.dataset_context.coco_file.annotations:
            image_id = annotation.image_id
            category_id = annotation.category_id
            image_categories[image_id] = category_id

        for image in self.dataset_context.coco_file.images:
            image_id = image.id
            if image_id in image_categories:
                category_id = image_categories[image_id]
                category_name = categories[category_id]

                category_dir = os.path.join(self.destination_dir, category_name)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)

                src_image_path = os.path.join(
                    self.dataset_context.image_dir, image.file_name
                )
                dest_image_path = os.path.join(category_dir, image.file_name)
                shutil.copy(src_image_path, dest_image_path)

        shutil.rmtree(self.dataset_context.image_dir)

        self.dataset_context.image_dir = self.dataset_context.dataset_extraction_path
        return self.dataset_context
