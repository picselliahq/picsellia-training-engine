from typing import List, Optional, Dict
import json

from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_context_validator import (
    DatasetContextValidator,
)


class CocoSegmentationDatasetContextValidator(
    DatasetContextValidator[CocoDatasetContext]
):
    """
    Validator for COCO Segmentation format annotations.
    """

    def __init__(self, dataset_context: CocoDatasetContext, fix_annotation=True):
        """
        Initialize the validator for a COCO segmentation dataset context.

        Args:
            dataset_context (CocoDatasetContext): The dataset context containing the COCO segmentation data to validate.
            fix_annotation (bool): A flag to indicate whether to automatically fix errors (default is True).

        Attributes:
            error_count (dict): A dictionary to track the count of different types of errors found during validation.
        """
        super().__init__(dataset_context=dataset_context, fix_annotation=fix_annotation)
        self.error_count = {
            "class_id": 0,
            "polygon_points": 0,
            "deleted_objects": 0,
        }

    def validate(self):
        """
        Validate the COCO segmentation dataset context.

        Ensures all COCO segmentation annotations are correctly formatted, within bounds, and valid.
        If any issues are found, they are reported, and optionally fixed.

        Returns:
            CocoDatasetContext: The validated (or fixed) dataset context.

        Raises:
            ValueError: If errors are detected in the annotations and `fix_annotation` is False.
        """
        super().validate()
        self._validate_labelmap()
        self._validate_coco_segmentation_annotations()
        if any(self.error_count.values()):
            self._report_errors()
        return self.dataset_context

    def _validate_labelmap(self):
        """
        Validate that the labelmap for the dataset context is valid.

        A valid COCO labelmap must contain at least one class.

        Raises:
            ValueError: If the labelmap does not contain at least one class.
        """
        if len(self.dataset_context.labelmap) < 1:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"A COCO labelmap must have at least 1 class."
            )

    def _validate_coco_segmentation_annotations(self):
        """
        Validate the COCO segmentation annotations in the dataset context.

        Ensures that all annotations are valid, well-formed, and within bounds. Invalid annotations
        are either fixed or deleted.

        Raises:
            ValueError: If no valid COCO data is found in the dataset context.
        """
        if not self.dataset_context.coco_data:
            raise ValueError(
                f"COCO not found for dataset {self.dataset_context.dataset_name}"
            )
        annotations = self.dataset_context.coco_data.get("annotations", [])
        updated_annotations = []

        for annotation in annotations:
            updated_annotation = self._validate_or_fix_annotation(annotation)
            if updated_annotation:
                updated_annotations.append(updated_annotation)
            else:
                # Annotation deleted due to invalid polygons or class_id
                self.error_count["deleted_objects"] += 1

        # Overwrite the annotations file if fix_annotation is enabled
        if self.fix_annotation:
            self.dataset_context.coco_data["annotations"] = updated_annotations
            if self.dataset_context.coco_file_path:
                with open(self.dataset_context.coco_file_path, "w") as file:
                    json.dump(self.dataset_context.coco_data, file, indent=4)
            else:
                print(
                    f'No COCO file path found for dataset "{self.dataset_context.dataset_name}, skipping saving.'
                )

    def _validate_or_fix_annotation(self, annotation: Dict) -> Optional[Dict]:
        """
        Validate or fix a single COCO segmentation annotation.

        Args:
            annotation (Dict): A COCO annotation object, which should contain a 'category_id' and 'segmentation'.

        Returns:
            Dict or None: The updated annotation object if valid, or None if it should be deleted due to invalid `category_id` or `segmentation`.
        """
        # Validate class_id
        class_id = annotation["category_id"]
        if class_id < 0 or class_id >= len(self.dataset_context.labelmap):
            self.error_count["class_id"] += 1
            print(
                f"Deleting annotation {annotation['id']} for image {annotation['image_id']}: "
                f"Invalid class_id {class_id}."
            )
            return None  # Delete the annotation if class_id is invalid

        # Validate and fix segmentation
        updated_segmentation = []
        for segmentation in annotation["segmentation"]:
            corrected_segmentation = self._validate_polygon(segmentation, annotation)
            if corrected_segmentation:
                updated_segmentation.append(corrected_segmentation)
            else:
                # Segmentation deleted
                self.error_count["deleted_objects"] += 1

        # If no valid segmentation remains, remove annotation
        if not updated_segmentation:
            return None

        # Update the segmentation
        if self.fix_annotation:
            annotation["segmentation"] = updated_segmentation

        return annotation

    def _validate_polygon(
        self, segmentation: List[float], annotation: dict
    ) -> Optional[List[float]]:
        """
        Validate or fix a single polygon segmentation.

        Args:
            segmentation (List[float]): The segmentation polygon points, alternating x and y coordinates.
            annotation (dict): The parent annotation object that contains the segmentation.

        Returns:
            List[float] or None: The corrected polygon points as a list of x and y coordinates, or None if the polygon is invalid.
        """
        object_has_error = False
        image = self._get_image_by_id(annotation["image_id"])

        corrected_segmentation = []
        for i in range(0, len(segmentation), 2):
            x, y = segmentation[i], segmentation[i + 1]

            # Validate x and y coordinates
            if not (0 <= x <= image["width"]):
                object_has_error = True
                if self.fix_annotation:
                    x = max(0, min(image["width"], x))

            if not (0 <= y <= image["height"]):
                object_has_error = True
                if self.fix_annotation:
                    y = max(0, min(image["height"], y))

            corrected_segmentation.extend([x, y])

        # If all x or y coordinates are identical, remove this polygon
        x_coords = corrected_segmentation[::2]
        y_coords = corrected_segmentation[1::2]

        if len(set(x_coords)) == 1 or len(set(y_coords)) == 1:
            print(
                f"Deleting polygon in annotation {annotation['id']} "
                f"for image {annotation['image_id']}: All x or y have the same value."
            )
            return None

        # If object has error, count it
        if object_has_error:
            self.error_count["polygon_points"] += 1

        return corrected_segmentation

    def _get_image_by_id(self, image_id: int) -> Dict:
        """
        Retrieve the image object by its ID from the dataset context.

        Args:
            image_id (int): The ID of the image to retrieve.

        Returns:
            Dict: The image object that corresponds to the given image ID.

        Raises:
            ValueError: If no image with the given ID is found in the dataset context.
        """
        if not self.dataset_context.coco_data:
            raise ValueError(
                f"COCO file not found for dataset {self.dataset_context.dataset_name}"
            )
        images = self.dataset_context.coco_data.get("images", [])
        for image in images:
            if image["id"] == image_id:
                return image
        raise ValueError(f"Image with ID {image_id} not found.")

    def _report_errors(self):
        """
        Report the errors found during the validation of COCO segmentation annotations.

        This method prints out the number of issues detected for each error type.
        """
        print(
            f"⚠️ Found {sum(self.error_count.values())} COCO segmentation annotation issues in dataset {self.dataset_context.dataset_name}:"
        )
        for error_type, count in self.error_count.items():
            print(f" - {error_type}: {count} issues")

        if self.fix_annotation:
            print("🔧 Fixing these issues automatically...")
        else:
            raise ValueError(
                "COCO segmentation annotation issues detected. Set 'fix_annotation' to True to automatically fix them."
            )
