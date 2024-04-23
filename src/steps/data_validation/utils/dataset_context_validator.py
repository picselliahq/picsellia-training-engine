from typing import List

from PIL import Image

from src.models.dataset.dataset_context import DatasetContext
from src.steps.data_validation.utils.image_utils import get_image_path_list


class DatasetContextValidator:
    """
    Validates various aspects of a dataset context.

    This class performs common validation tasks for dataset contexts, including checking for image extraction
    completeness, image format, image corruption, and annotation integrity.

    Attributes:
        dataset_context (DatasetContext): The dataset context to validate.
    """

    VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, dataset_context: DatasetContext):
        """
        Initializes the DatasetContextValidator with a dataset context to validate.

        Parameters:
            dataset_context (DatasetContext): The dataset context to validate.
        """
        self.dataset_context = dataset_context

    def validate(self):
        """
        Validates the dataset context.
        """
        if self.dataset_context.image_dir:
            image_path_list = get_image_path_list(
                image_dir=self.dataset_context.image_dir
            )
            self.validate_image_extraction(image_path_list=image_path_list)
            self.validate_image_format(image_path_list=image_path_list)
            self.validate_image_corruption(image_path_list=image_path_list)
        else:
            raise ValueError(
                f"Image directory is missing from the dataset context in {self.dataset_context.dataset_name} dataset"
            )

    def validate_image_extraction(self, image_path_list: List[str]) -> None:
        """
        Validates that the number of extracted images matches the expected number of assets.

        Args:
            image_path_list (List[str]): The list of image paths extracted from the dataset.

        Raises:
            ValueError: If the number of extracted images does not match the expected number of assets.
        """
        if len(image_path_list) < len(self.dataset_context.multi_asset):
            raise ValueError(
                f"Some images have not been extracted in {self.dataset_context.dataset_name} dataset"
            )
        if len(image_path_list) > len(self.dataset_context.multi_asset):
            raise ValueError(
                f"There are more images than expected in {self.dataset_context.dataset_name} dataset"
            )

    def validate_image_format(self, image_path_list: List[str]) -> None:
        """
        Validates that all images in the dataset are in an expected format.

        Args:
            image_path_list (List[str]): The list of image paths extracted from the dataset.

        Raises:
            ValueError: If any image is not in one of the valid formats.
        """
        for image_path in image_path_list:
            if not image_path.lower().endswith(self.VALID_IMAGE_EXTENSIONS):
                raise ValueError(
                    f"Invalid image format for image {image_path} in {self.dataset_context.dataset_name} dataset. "
                    f"Valid image formats are {self.VALID_IMAGE_EXTENSIONS}"
                )

    def validate_image_corruption(self, image_path_list: List[str]) -> None:
        """
        Checks for corruption in the extracted images.

        Parameters:
            image_path_list (List[str]): The list of image paths extracted from the dataset.

        Raises:
            ValueError: If any of the images are found to be corrupted.
        """
        for image_path in image_path_list:
            try:
                with Image.open(image_path) as img:
                    img.verify()  # Verify that this is a valid image
            except Exception as e:
                raise ValueError(
                    f"Image {image_path} is corrupted in {self.dataset_context.dataset_name} dataset and cannot be used."
                ) from e
