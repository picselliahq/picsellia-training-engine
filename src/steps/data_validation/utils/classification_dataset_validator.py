import os
from abc import ABC

from src.steps.data_validation.utils.dataset_validator import DatasetValidator


class ClassificationDatasetValidator(DatasetValidator, ABC):
    def validate(self):
        super().validate()  # Call common validations
        self.validate_labelmap()
        self.validate_at_least_one_image_per_class()

    def validate_labelmap(self):
        for dataset_context in self.dataset_collection:
            if len(dataset_context.labelmap) < 2:
                raise ValueError(
                    f"Labelmap for dataset {dataset_context.name} is not valid. "
                    f"A classification labelmap must have at least 2 classes. "
                    f"Current labelmap is {dataset_context.labelmap}"
                )

    def validate_at_least_one_image_per_class(self):
        class_names = self.dataset_collection.train.labelmap.keys()
        for class_name in class_names:
            folder_path = os.path.join(
                self.dataset_collection.train.image_dir, str(class_name)
            )
            if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
                raise ValueError(
                    f"Train dataset must have at least 1 image for class {class_name}"
                )
