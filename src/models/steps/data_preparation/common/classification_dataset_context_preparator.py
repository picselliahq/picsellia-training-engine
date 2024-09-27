import os
import shutil
from typing import Dict

from picsellia_annotations.coco import Image
from src.models.dataset.common.dataset_context import DatasetContext


class ClassificationDatasetContextPreparator:
    """
    Prepares and organizes dataset images into directories based on their classification categories.

    This class takes a dataset context with category and annotation information in COCO format.
    It organizes the dataset by creating a directory for each category and moves the images into their
    respective category directories, which is often required for classification tasks in deep learning frameworks.

    Attributes:
        dataset_context (DatasetContext): The context of the dataset including paths and COCO file.
        destination_path (str): The target directory where the images will be moved and organized.
    """

    def __init__(self, dataset_context: DatasetContext, destination_path: str):
        """
        Initializes the preparator with a given dataset context and a destination directory for images.

        Args:
            dataset_context (DatasetContext): The context of the dataset to organize.
            destination_path (str): The directory where the organized images will be stored.

        Raises:
            ValueError: If the destination image directory is the same as the original image directory.
        """
        self.dataset_context = dataset_context
        self.destination_path = destination_path
        if self.dataset_context.images_dir == self.destination_path:
            raise ValueError(
                "The destination image directory cannot be the same as the original image directory."
            )
        if not self.dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")

    def organize(self) -> DatasetContext:
        """
        Organizes the dataset by creating category directories and moving images.

        Extracts category information from the COCO file, maps images to their categories,
        and organizes the images into the respective category directories. Cleans up the original
        image directory and annotations directory after moving the images.

        Returns:
            DatasetContext: The updated dataset context with the new image directory.
        """
        categories = self._extract_categories()
        image_categories = self._map_image_to_category()
        self._organize_images(categories, image_categories)

        # Remove the old images directory once images are moved
        if not self.dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        shutil.rmtree(self.dataset_context.images_dir)
        self.dataset_context.images_dir = self.destination_path

        return self.dataset_context

    def _extract_categories(self) -> Dict[int, str]:
        """
        Extracts the categories from the dataset's COCO file.

        Returns:
            Dict[int, str]: A dictionary mapping category IDs to category names.
        """
        if not self.dataset_context.coco_file:
            raise ValueError("No COCO file found in the dataset context.")
        return {
            category.id: category.name
            for category in self.dataset_context.coco_file.categories
        }

    def _map_image_to_category(self) -> Dict[int, int]:
        """
        Maps each image to its category based on the annotations in the COCO file.

        Returns:
            Dict[int, int]: A dictionary mapping image IDs to category IDs.
        """
        if not self.dataset_context.coco_file:
            raise ValueError("No COCO file found in the dataset context.")
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
        if not self.dataset_context.coco_file:
            raise ValueError("No COCO file found in the dataset context.")
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
        Creates a directory for a category if it doesn't exist and moves an image into it.

        Args:
            category_name (str): The name of the category.
            image (Image): The image object containing file name and metadata.

        Raises:
            PermissionError: If there is a permission issue when creating the directory or moving the file.
            FileNotFoundError: If the source image file is not found.
            shutil.SameFileError: If the source and destination paths are the same.
        """
        category_dir = os.path.join(self.destination_path, category_name)
        os.makedirs(category_dir, exist_ok=True)
        if not self.dataset_context.images_dir:
            raise ValueError("No images directory found in the dataset context.")
        src_image_path = os.path.join(self.dataset_context.images_dir, image.file_name)
        dest_image_path = os.path.join(category_dir, image.file_name)
        shutil.move(src_image_path, dest_image_path)
