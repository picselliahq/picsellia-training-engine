import os
from abc import abstractmethod, ABC
from typing import List

from PIL import Image

from src.models.dataset.dataset_collection import DatasetCollection
from src.models.dataset.dataset_context import DatasetContext


def get_image_path_list(image_dir: str) -> List[str]:
    """
    Generates a list of all image file paths within a specified directory.

    Args:
        image_dir (str): The directory to search for image files.

    Returns:
        List[str]: A list containing the paths to all images found within the directory and its subdirectories.
    """
    image_path_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path_list.append(os.path.join(root, file))
    return image_path_list


def validate_image_extraction(
    dataset_context: DatasetContext, image_path_list: List[str]
) -> None:
    """
    Validates that the number of extracted images matches the expected number of assets.

    Args:
        dataset_context (DatasetContext): The dataset context to validate.
        image_path_list (List[str]): The list of image paths extracted from the dataset.

    Raises:
        ValueError: If the number of extracted images does not match the expected number of assets.
    """
    if len(image_path_list) < len(dataset_context.multi_asset):
        raise ValueError(
            f"Some images have not been extracted in {dataset_context.dataset_name} dataset"
        )
    if len(image_path_list) > len(dataset_context.multi_asset):
        raise ValueError(
            f"There are more images than expected in {dataset_context.dataset_name} dataset"
        )


def validate_image_corruption(
    dataset_context: DatasetContext, image_path_list: List[str]
) -> None:
    """
    Checks for corruption in the extracted images.

    Parameters:
        dataset_context (DatasetContext): The dataset context to validate.
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
                f"Image {image_path} is corrupted in {dataset_context.dataset_name} dataset and cannot be used."
            ) from e


class DatasetValidator(ABC):
    """
    Validates various aspects of datasets within a dataset collection.

    This class performs common validation tasks for datasets, including checking for image extraction
    completeness, image format, image corruption, and annotation integrity. It serves as a base class
    for more specific dataset validators.

    Attributes:
        dataset_collection (DatasetCollection): The collection of datasets to validate.
    """

    VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, dataset_collection: DatasetCollection):
        """
        Initializes the DatasetValidator with a dataset collection to validate.

        Parameters:
            dataset_collection (DatasetCollection): The dataset collection to validate.
        """
        self.dataset_collection = dataset_collection

    def validate_common(self) -> None:
        """
        Performs common validation tasks across all datasets in the collection.
        """
        for dataset_context in self.dataset_collection:
            image_path_list = get_image_path_list(image_dir=dataset_context.image_dir)
            validate_image_extraction(
                dataset_context=dataset_context, image_path_list=image_path_list
            )
            self.validate_image_format(
                dataset_context=dataset_context, image_path_list=image_path_list
            )
            validate_image_corruption(
                dataset_context=dataset_context, image_path_list=image_path_list
            )
            self.validate_image_annotation_integrity(dataset_context)

    def validate_image_format(
        self, dataset_context: DatasetContext, image_path_list: List[str]
    ) -> None:
        """
        Validates that all images in the dataset are in an expected format.

        Args:
            dataset_context (DatasetContext): The dataset context to validate.
            image_path_list (List[str]): The list of image paths extracted from the dataset.

        Raises:
            ValueError: If any image is not in one of the valid formats.
        """
        for image_path in image_path_list:
            if not image_path.endswith(self.VALID_IMAGE_EXTENSIONS):
                raise ValueError(
                    f"Invalid image format for image {image_path} in {dataset_context.dataset_name} dataset. "
                    f"Valid image formats are {self.VALID_IMAGE_EXTENSIONS}"
                )

    @abstractmethod
    def validate_image_annotation_integrity(
        self, dataset_context: DatasetContext
    ) -> None:
        """
        Placeholder for validating the integrity of image annotations. This should be implemented by subclasses.

        Parameters:
            dataset_context (DatasetContext): The dataset context to validate.
        """
        pass

    @abstractmethod
    def validate(self):
        """
        Validates the dataset collection. This method is intended to be overridden by subclasses to include
        specific validation logic in addition to the common validations.
        """
        self.validate_common()
