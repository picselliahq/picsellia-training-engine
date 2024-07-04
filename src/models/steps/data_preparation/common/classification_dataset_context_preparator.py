import os
import shutil
from typing import Dict

from picsellia_annotations.coco import Image
from src.models.dataset.common.dataset_context import DatasetContext


class ClassificationDatasetContextPreparator:
    """
    Organizes dataset images into directories based on their classification categories.

    This class takes a dataset context which includes a COCO file with category and annotation information.
    It organizes the dataset by creating a directory for each category and moving the images into their respective
    category directories. This structure is often required for many deep learning frameworks and simplifies the
    task of dataset loading for classification tasks.

    Attributes:
        dataset_context (DatasetContext): The context of the dataset to organize, including paths and COCO file.
        destination_path (str): The root directory where the organized dataset will be stored.
    """

    def __init__(self, dataset_context: DatasetContext):
        """
        Initializes the organizer with a given dataset context.

        Args:
            dataset_context (DatasetContext): The dataset context to organize.
        """
        self.dataset_context = dataset_context
        self.destination_path = dataset_context.destination_path

    def organize(self) -> None:
        """
        Organizes the dataset by creating category directories and moving images.

        Extracts category information from the COCO file, maps images to their categories,
        and organizes the images into the respective category directories. Finally, cleans up
        the original image directory.
        """
        categories = self._extract_categories()
        image_categories = self._map_image_to_category()
        self._organize_images(categories, image_categories)
        self._cleanup()

    def _extract_categories(self) -> Dict[int, str]:
        """
        Extracts the categories from the dataset's COCO file.

        Returns:
            - Dict[int, str]: A dictionary mapping category IDs to category names.
        """
        return {
            category.id: category.name
            for category in self.dataset_context.coco_file.categories
        }

    def _map_image_to_category(self) -> Dict[int, int]:
        """
        Maps each image to its category based on annotations in the COCO file.

        Returns:
            - Dict[int, int]: A dictionary mapping image IDs to category IDs.
        """
        return {
            annotation.image_id: annotation.category_id
            for annotation in self.dataset_context.coco_file.annotations
        }

    def _organize_images(
        self, categories: Dict[int, str], image_categories: Dict[int, int]
    ) -> None:
        """
        Creates category directories and moves images into their respective directories.

        Args:
            categories (Dict[int, str]): A mapping from category IDs to category names.
            image_categories (Dict[int, int]): A mapping from image IDs to category IDs.
        """
        for image in self.dataset_context.coco_file.images:
            image_id = image.id
            if image_id in image_categories:
                category_id = image_categories[image_id]
                category_name = categories[category_id]
                self._create_category_dir_and_copy_image(category_name, image)

    def _create_category_dir_and_copy_image(
        self, category_name: str, image: Image
    ) -> None:
        """
        Creates a directory for a category if it doesn't exist and copies an image into it.

        Parameters:
            category_name (str): The name of the category.
            image: The image object containing the file name.

        Raises:
            PermissionError: If there is a permission issue creating the directory or copying the file.
            FileNotFoundError: If the source image file to copy is not found.
            shutil.SameFileError: If the source and destination paths are the same.
        """
        category_dir = os.path.join(self.destination_path, category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        src_image_path = os.path.join(self.dataset_context.image_dir, image.file_name)
        dest_image_path = os.path.join(category_dir, image.file_name)
        shutil.copy(src_image_path, dest_image_path)

    def _cleanup(self) -> None:
        """
        Cleans up the original image directory after organizing the images.

        Removes the original image directory and updates the `image_dir` attribute in the dataset context
        to point to the new, organized directory structure.
        """
        shutil.rmtree(self.dataset_context.image_dir)
        self.dataset_context.image_dir = self.destination_path
