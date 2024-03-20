import os
import shutil


class DatasetOrganizer:
    def __init__(self, coco_file, images_dir, dest_dir):
        self.coco_file = coco_file
        self.images_dir = images_dir
        self.dest_dir = dest_dir

    def organize(self):
        categories = {
            category.id: category.name for category in self.coco_file.categories
        }
        image_categories = {}
        for annotation in self.coco_file.annotations:
            image_id = annotation.image_id
            category_id = annotation.category_id
            image_categories[image_id] = category_id

        for image in self.coco_file.images:
            image_id = image.id
            if image_id in image_categories:
                category_id = image_categories[image_id]
                category_name = categories[category_id]

                category_dir = os.path.join(self.dest_dir, category_name)
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)

                src_image_path = os.path.join(self.images_dir, image.file_name)
                dest_image_path = os.path.join(category_dir, image.file_name)
                shutil.copy(src_image_path, dest_image_path)
