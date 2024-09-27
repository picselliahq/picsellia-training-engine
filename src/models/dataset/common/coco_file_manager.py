from typing import List, Dict, Optional
from collections import defaultdict

from picsellia_annotations.coco import COCOFile, Image, Annotation


class COCOFileManager:
    """
    A class to manage and facilitate the access of information in a COCO file.

    This class provides methods to retrieve images, annotations, categories, and other
    metadata from a COCO file. It also builds indices for faster lookup of these elements.
    """

    def __init__(self, coco_file: COCOFile):
        """
        Initializes the COCOFileManager with a given COCO file.

        Args:
            coco_file (COCOFile): The COCO file containing dataset information.
        """
        self.coco_file = coco_file
        self._build_indices()

    def _build_indices(self):
        """
        Builds indices to speed up lookups for categories, images, and annotations.

        This method creates mappings for faster retrieval of category names, image file names,
        and annotations based on image IDs or category IDs.
        """
        self.category_id_to_name = {
            cat.id: cat.name for cat in self.coco_file.categories
        }
        self.category_name_to_id = {
            cat.name: cat.id for cat in self.coco_file.categories
        }
        self.image_id_to_filename = {
            img.id: img.file_name for img in self.coco_file.images
        }
        self.filename_to_image_id = {
            img.file_name: img.id for img in self.coco_file.images
        }

        self.image_id_to_annotations = defaultdict(list)
        for ann in self.coco_file.annotations:
            self.image_id_to_annotations[ann.image_id].append(ann)

    def get_category_name(self, category_id: int) -> Optional[str]:
        """
        Returns the category name for a given category ID.

        Args:
            category_id (int): The ID of the category.

        Returns:
            Optional[str]: The name of the category, or None if not found.
        """
        return self.category_id_to_name.get(category_id)

    def get_category_id(self, category_name: str) -> Optional[int]:
        """
        Returns the category ID for a given category name.

        Args:
            category_name (str): The name of the category.

        Returns:
            Optional[int]: The ID of the category, or None if not found.
        """
        return self.category_name_to_id.get(category_name)

    def get_image_filename(self, image_id: int) -> Optional[str]:
        """
        Returns the filename of an image given its ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            Optional[str]: The filename of the image, or None if not found.
        """
        return self.image_id_to_filename.get(image_id)

    def get_image_id(self, filename: str) -> Optional[int]:
        """
        Returns the image ID for a given filename.

        Args:
            filename (str): The filename of the image.

        Returns:
            Optional[int]: The ID of the image, or None if not found.
        """
        return self.filename_to_image_id.get(filename)

    def get_annotations_for_image(self, image_id: int) -> List[Annotation]:
        """
        Returns all annotations associated with a given image.

        Args:
            image_id (int): The ID of the image.

        Returns:
            List[Annotation]: A list of annotations associated with the image.
        """
        return self.image_id_to_annotations.get(image_id, [])

    def get_images_for_category(self, category_id: int) -> List[Image]:
        """
        Returns all images that contain a given category.

        Args:
            category_id (int): The ID of the category.

        Returns:
            List[Image]: A list of images that contain the specified category.
        """
        image_ids = set(
            ann.image_id
            for ann in self.coco_file.annotations
            if ann.category_id == category_id
        )
        return [img for img in self.coco_file.images if img.id in image_ids]

    def get_annotation_count_per_category(self) -> Dict[str, int]:
        """
        Returns the number of annotations per category.

        This method counts how many annotations are present for each category in the COCO dataset.

        Returns:
            Dict[str, int]: A dictionary mapping category names to the number of annotations.
        """
        count: defaultdict[str, int] = defaultdict(int)
        for ann in self.coco_file.annotations:
            category_name = self.get_category_name(ann.category_id)
            if category_name is not None:
                count[category_name] += 1
        return dict(count)

    def get_image_dimensions(self, image_id: int) -> Optional[Dict[str, int]]:
        """
        Returns the dimensions (width and height) of an image given its ID.

        Args:
            image_id (int): The ID of the image.

        Returns:
            Optional[Dict[str, int]]: A dictionary with the width and height of the image, or None if not found.
        """
        for img in self.coco_file.images:
            if img.id == image_id:
                return {"width": img.width, "height": img.height}
        return None

    def get_annotations_by_category(self, category_id: int) -> List[Annotation]:
        """
        Returns all annotations for a given category.

        Args:
            category_id (int): The ID of the category.

        Returns:
            List[Annotation]: A list of annotations associated with the specified category.
        """
        return [
            ann for ann in self.coco_file.annotations if ann.category_id == category_id
        ]
