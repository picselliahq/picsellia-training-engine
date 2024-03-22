import os

from PIL import Image

from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from src.models.dataset.dataset_type import DatasetType


def get_image_path_list(image_dir) -> list:
    image_path_list = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            image_path_list.append(os.path.join(root, file))
    return image_path_list


def _validate_image_extraction(dataset_context, image_path_list):
    if len(image_path_list) < len(dataset_context.multi_asset):
        raise ValueError(
            f"Some images have not been extracted in {dataset_context.name} dataset"
        )
    if len(image_path_list) > len(dataset_context.multi_asset):
        raise ValueError(
            f"There are more images than expected in {dataset_context.name} dataset"
        )


def _validate_image_corruption(dataset_context, image_path_list):
    for image_path in image_path_list:
        try:
            with Image.open(image_path) as img:
                img.verify()  # Verify that this is a valid image
        except Exception as e:
            raise ValueError(
                f"Image {image_path} is corrupted in {dataset_context.name} dataset - {e}"
            )


class DatasetValidator:
    VALID_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

    def __init__(self, dataset_collection: DatasetCollection):
        self.dataset_collection = dataset_collection

    def validate_common(self):
        for dataset_name in (
            DatasetType.TRAIN.value,
            DatasetType.VAL.value,
            DatasetType.TEST.value,
        ):
            dataset_context = getattr(self.dataset_collection, dataset_name)
            image_path_list = get_image_path_list(dataset_context.image_dir)
            _validate_image_extraction(dataset_context, image_path_list)
            self._validate_image_format(dataset_context, image_path_list)
            _validate_image_corruption(dataset_context, image_path_list)
            self._validate_image_annotation_integrity(dataset_context)

    def _validate_image_format(self, dataset_context, image_path_list):
        for image_path in image_path_list:
            if not image_path.endswith(self.VALID_IMAGE_EXTENSIONS):
                raise ValueError(
                    f"Invalid image format for image {image_path} in {dataset_context.name} dataset"
                )

    def _validate_image_annotation_integrity(self, dataset_context):
        pass

    def validate(self):
        # This method will be overridden by subclasses
        self.validate_common()


class ClassificationDatasetValidator(DatasetValidator):
    def validate(self):
        super().validate()  # Call common validations
        self._validate_labelmap()
        self._validate_at_least_one_image_per_class()

    def _validate_labelmap(self):
        for dataset_name in (
            DatasetType.TRAIN.value,
            DatasetType.VAL.value,
            DatasetType.TEST.value,
        ):
            dataset_context = getattr(self.dataset_collection, dataset_name)
            if len(dataset_context.labelmap) < 2:
                raise ValueError(
                    f"Labelmap must have at least 2 classes in {dataset_context.name} dataset"
                )

    def _validate_at_least_one_image_per_class(self):
        class_names = self.dataset_collection.train.labelmap.keys()
        for class_name in class_names:
            folder_path = os.path.join(
                self.dataset_collection.train.image_dir, str(class_name)
            )
            if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
                raise ValueError(
                    f"Train dataset must have at least 1 image for class {class_name}"
                )
