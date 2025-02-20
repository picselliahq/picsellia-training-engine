from typing import List, Optional
import os

from src.picsellia_cv_engine.models.dataset.yolo_dataset_context import (
    YoloDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_context_validator import (
    DatasetContextValidator,
)


class YoloSegmentationDatasetContextValidator(
    DatasetContextValidator[YoloDatasetContext]
):
    """
    Validator for YOLO Segmentation format annotations.
    """

    def __init__(self, dataset_context: YoloDatasetContext, fix_annotation=True):
        """
        Initializes the YOLO segmentation dataset context validator.

        Args:
            dataset_context (YoloDatasetContext): The context of the YOLO dataset containing annotation data.
            fix_annotation (bool): Flag to indicate whether to automatically fix the detected issues.
        """
        super().__init__(dataset_context=dataset_context, fix_annotation=fix_annotation)
        self.error_count = {
            "class_id": 0,
            "polygon_points": 0,
            "deleted_objects": 0,
        }

    def validate(self):
        """
        Validates the YOLO segmentation dataset context.

        This method checks whether the annotation files are correctly formatted,
        validates class IDs and polygon points, and optionally fixes issues.
        It also reports the number of errors detected during validation.

        Returns:
            YoloDatasetContext: The validated or updated dataset context.

        Raises:
            ValueError: If any errors are found and `fix_annotation` is set to False.
        """
        super().validate()
        self._validate_labelmap()
        self._validate_yolo_segmentation_annotations()
        if any(self.error_count.values()):
            self._report_errors()
        return self.dataset_context

    def _validate_labelmap(self):
        """
        Validates the labelmap of the dataset context.

        Ensures that the YOLO dataset has at least one class in its labelmap.

        Raises:
            ValueError: If the labelmap is empty or contains no classes.
        """
        if len(self.dataset_context.labelmap) < 1:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"A YOLO labelmap must have at least 1 class."
            )

    def _validate_yolo_segmentation_annotations(self):
        """
        Validates YOLO segmentation annotation files.

        Iterates over the annotation directory and checks each `.txt` file containing
        YOLO segmentation annotations. It ensures that all annotations are valid,
        and automatically fixes any issues if `fix_annotation` is set to True.

        If an annotation file becomes completely invalid, it is deleted.

        Raises:
            ValueError: If the annotations directory does not exist or is missing.
        """
        annotations_dir = self.dataset_context.annotations_dir
        if not annotations_dir or not os.path.exists(annotations_dir):
            raise ValueError(
                f"Annotations directory is missing for dataset {self.dataset_context.dataset_name}."
            )

        for annotation_file in os.listdir(annotations_dir):
            if not annotation_file.endswith(".txt"):
                continue
            annotation_path = os.path.join(annotations_dir, annotation_file)
            with open(annotation_path, "r") as file:
                lines = file.readlines()
            updated_lines = self._validate_annotation_file(
                lines=lines, annotation_file=annotation_file
            )

            # Overwrite the annotation file with updated lines
            if updated_lines:
                with open(annotation_path, "w") as file:
                    file.writelines(updated_lines)
            else:
                print(
                    f"Deleting file {annotation_path} because all its lines were invalid."
                )
                try:
                    os.remove(annotation_path)
                except OSError as e:
                    print(f"Error deleting file {annotation_path}: {e}")

    def _validate_annotation_file(
        self, lines: List[str], annotation_file: str
    ) -> List[str]:
        """
        Validates a single YOLO segmentation annotation file.

        This method processes each line in the file, validates the class ID and polygon points,
        and attempts to fix errors when possible. Invalid lines are discarded, and valid ones
        are returned for updating the file.

        Args:
            lines (List[str]): The lines from the annotation file to validate.
            annotation_file (str): The name of the annotation file being processed.

        Returns:
            List[str]: A list of updated lines for the annotation file, or an empty list if the file is completely invalid.
        """
        updated_lines = []

        for line_num, line in enumerate(lines, start=1):
            try:
                try:
                    fields = line.strip().split()
                    class_id = int(fields[0])
                    polygon_points = list(map(float, fields[1:]))
                except ValueError:
                    print(f"Skipping invalid line {line_num} in {annotation_file}")
                    continue

                if len(polygon_points) % 2 != 0:
                    raise ValueError(
                        f"Line {line_num} in {annotation_file} does not have an even number of polygon points."
                    )

                updated_line = self._validate_or_fix_annotation(
                    class_id=class_id,
                    polygon_points=polygon_points,
                    line_num=line_num,
                    annotation_file=annotation_file,
                )
                if updated_line:
                    updated_lines.append(updated_line)
                else:
                    # Object deleted due to invalid x or y
                    self.error_count["deleted_objects"] += 1

            except ValueError as e:
                raise ValueError(
                    f"Error in line {line_num} of {annotation_file}: {str(e)}"
                )

        return updated_lines

    def _validate_or_fix_annotation(
        self,
        class_id: int,
        polygon_points: List[float],
        line_num: int,
        annotation_file: str,
    ) -> Optional[str]:
        """
        Validates or fixes a single YOLO segmentation annotation.

        This method checks the class ID and polygon points, and if any issues are found,
        it either fixes them (if `fix_annotation` is enabled) or deletes the annotation
        by returning None.

        If all points are invalid (e.g., all points are 0), the annotation is discarded.

        Args:
            class_id (int): The class ID of the object in the annotation.
            polygon_points (List[float]): The list of normalized (x, y) coordinates of the polygon.
            line_num (int): The line number in the annotation file.
            annotation_file (str): The name of the annotation file.

        Returns:
            str: The updated annotation line as a string, or None if the annotation is invalid.
        """
        object_has_error = False  # Track if this object has at least one error

        # Validate class_id
        if class_id < 0 or class_id >= len(self.dataset_context.labelmap):
            print(
                f"Deleting object in {annotation_file} on line {line_num} due to invalid class_id {class_id}"
            )
            return None

        corrected_points = []
        for i in range(0, len(polygon_points), 2):
            x, y = polygon_points[i], polygon_points[i + 1]

            # Check x-coordinate
            if not (0 <= x <= 1):
                object_has_error = True
                if self.fix_annotation:
                    x = max(0, min(1, x))

            # Check y-coordinate
            if not (0 <= y <= 1):
                object_has_error = True
                if self.fix_annotation:
                    y = max(0, min(1, y))

            corrected_points.extend([x, y])

        # Check if all x or all y are 0
        x_coords = corrected_points[::2]
        y_coords = corrected_points[1::2]

        if len(set(x_coords)) == 1 or len(set(y_coords)) == 1:
            print(
                f"Deleting object in line {line_num} of {annotation_file}: "
                f"All x or all y have the same value."
            )
            return None

        # If the object had any errors, count it as one error
        if object_has_error:
            print(f"Correcting object in line {line_num} of {annotation_file}")
            self.error_count["polygon_points"] += 1

        # Return updated line
        return f"{class_id} " + " ".join(f"{p:.6f}" for p in corrected_points) + "\n"

    def _report_errors(self):
        """
        Reports the validation errors found in the dataset annotations.

        This method prints a summary of the errors detected during the validation process,
        including the number of issues related to class IDs, polygon points, and deleted objects.

        If `fix_annotation` is enabled, it will attempt to automatically fix these issues,
        otherwise, it raises a `ValueError`.
        """
        print(
            f"⚠️ Found {sum(self.error_count.values())} YOLO segmentation annotation issues in dataset {self.dataset_context.dataset_name}:"
        )
        for error_type, count in self.error_count.items():
            print(f" - {error_type}: {count} issues")

        if self.fix_annotation:
            print("🔧 Fixing these issues automatically...")
        else:
            raise ValueError(
                "YOLO segmentation annotation issues detected. Set 'fix_annotation' to True to automatically fix them."
            )
