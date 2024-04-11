import os
from abc import ABC

from src.steps.data_validation.utils.dataset_validator import DatasetValidator


class ClassificationDatasetValidator(DatasetValidator, ABC):
    def validate(self):
        """
        Validate the classification dataset.
        A classification dataset must have at least 2 classes and at least 1 image per class

        Raises:
            ValueError: If the classification dataset is not valid.
        """
        super().validate()  # Call common validations
        self.validate_labelmap()
        self.validate_at_least_one_image_per_class()

    def validate_labelmap(self):
        """
        Validate that the labelmap for each dataset in the collection is valid.
        A classification labelmap must have at least 2 classes.

        Raises:
            ValueError: If the labelmap for any dataset in the collection is not valid.
        """
        for dataset_context in self.dataset_collection:
            if len(dataset_context.labelmap) < 2:
                raise ValueError(
                    f"Labelmap for dataset {dataset_context.dataset_name} is not valid. "
                    f"A classification labelmap must have at least 2 classes. "
                    f"Current labelmap is {dataset_context.labelmap}"
                )

    def validate_at_least_one_image_per_class(self):
        """
        Validate that each class in the classification dataset has at least 1 image. If a class has no images,
        the dataset is considered invalid.

        Raises:
            ValueError: If any class in any dataset in the collection has no images.

        """
        for dataset_context in self.dataset_collection:
            class_names = dataset_context.labelmap.keys()
            for class_name in class_names:
                folder_path = os.path.join(
                    self.dataset_collection.train.image_dir, str(class_name)
                )
                if not os.path.exists(folder_path) or len(os.listdir(folder_path)) == 0:
                    raise ValueError(
                        f"{dataset_context.dataset_name} dataset must have at least 1 image for class {class_name}"
                    )
