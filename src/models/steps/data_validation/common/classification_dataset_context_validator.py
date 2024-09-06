import json
from collections import defaultdict
from typing import Dict

from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetContextValidator,
)


class ClassificationDatasetContextValidator(DatasetContextValidator):
    def validate(self):
        """
        Validate the classification dataset context.
        A classification dataset context must have at least 2 classes and at least 1 image per class

        Raises:
            ValueError: If the classification dataset context is not valid.
        """
        super().validate()  # Call common validations
        self.validate_labelmap()
        self.validate_coco_file()

        return self.dataset_context

    def validate_labelmap(self):
        """
        Validate that the labelmap for the dataset context is valid.
        A classification labelmap must have at least 2 classes.

        Raises:
            ValueError: If the labelmap for the dataset context is not valid.
        """
        if len(self.dataset_context.labelmap) < 2:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"A classification labelmap must have at least 2 classes. "
                f"Current labelmap is {self.dataset_context.labelmap}"
            )

    def validate_coco_file(self):
        """
        Validate that each class in the classification dataset has at least 1 image using the COCO file.
        If a class has no images, the dataset is considered invalid.

        Raises:
            ValueError: If any class in the classification dataset context has no images.
            FileNotFoundError: If the COCO file is not found.
            json.JSONDecodeError: If the COCO file is not a valid JSON.
        """
        try:
            with open(self.dataset_context.coco_file_path, "r") as f:
                coco_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"COCO file not found at {self.dataset_context.coco_file_path}"
            )
        except json.JSONDecodeError:
            raise ValueError(
                f"Invalid JSON in COCO file at {self.dataset_context.coco_file_path}"
            )

        # Create a mapping of category_id to category_name
        category_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

        # Count images per category
        images_per_category: Dict[str, int] = defaultdict(int)

        for annotation in coco_data["annotations"]:
            category_name = category_map[annotation["category_id"]]
            images_per_category[category_name] += 1

        # Check if each class in the labelmap has at least one image
        for class_name in self.dataset_context.labelmap.keys():
            if images_per_category[class_name] == 0:
                raise ValueError(
                    f"{self.dataset_context.dataset_name} dataset must have at least 1 image for class {class_name}"
                )

        # Check if there are any classes in the COCO file that are not in the labelmap
        coco_classes = set(category_map.values())
        labelmap_classes = set(self.dataset_context.labelmap.keys())
        extra_classes = coco_classes - labelmap_classes
        if extra_classes:
            raise ValueError(
                f"The following classes are present in the COCO file but not in the labelmap: {extra_classes}"
            )
