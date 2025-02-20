from typing import List
import os

from src.picsellia_cv_engine.models.dataset.yolo_dataset_context import (
    YoloDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_context_validator import (
    DatasetContextValidator,
)


class YoloObjectDetectionDatasetContextValidator(
    DatasetContextValidator[YoloDatasetContext]
):
    """
    Validator for YOLO format annotations.
    """

    def __init__(self, dataset_context: YoloDatasetContext, fix_annotation=True):
        """
        Initializes the YOLO object detection dataset context validator.

        Args:
            dataset_context (YoloDatasetContext): The context object containing dataset information and annotations.
            fix_annotation (bool): Flag indicating whether to automatically fix the detected issues.
        """
        super().__init__(dataset_context=dataset_context, fix_annotation=fix_annotation)
        self.error_count = {
            "class_id": 0,
            "x_center": 0,
            "y_center": 0,
            "width": 0,
            "height": 0,
        }

    def validate(self):
        """
        Validates the YOLO object detection dataset context.

        This method performs the following validation checks:
        - Verifies that the labelmap contains at least one class.
        - Validates the format and bounds of the YOLO annotations.
        - Reports any issues found during the validation process.

        Returns:
            YoloDatasetContext: The validated or updated dataset context.

        Raises:
            ValueError: If validation errors are found and `fix_annotation` is set to False.
        """
        super().validate()
        self._validate_labelmap()
        self._validate_yolo_annotations()
        if any(self.error_count.values()):
            self._report_errors()
        return self.dataset_context

    def _validate_labelmap(self):
        """
        Validates the labelmap of the dataset context.

        Ensures that the labelmap contains at least one class ID.

        Raises:
            ValueError: If the labelmap is empty or contains no classes.
        """
        if len(self.dataset_context.labelmap) < 1:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"A YOLO labelmap must have at least 1 class."
            )

    def _validate_yolo_annotations(self):
        """
        Validates YOLO annotations in the dataset.

        This method checks the annotations in the directory, ensuring that each annotation file is valid.
        It also corrects issues based on the `fix_annotation` flag.

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
            self._validate_annotation_file(lines=lines, annotation_file=annotation_file)

    def _validate_annotation_file(self, lines: List[str], annotation_file: str):
        """
        Validates an individual YOLO annotation file.

        This method processes each line in the annotation file, checking that all five fields (class ID,
        x_center, y_center, width, and height) are properly formatted and within valid bounds.

        Args:
            lines (List[str]): The lines from the annotation file to validate.
            annotation_file (str): The name of the annotation file.

        Raises:
            ValueError: If an annotation line is invalid.
        """
        for line_num, line in enumerate(lines, start=1):
            try:
                fields = list(map(float, line.strip().split()))
                if len(fields) != 5:
                    raise ValueError(
                        f"Line {line_num} in {annotation_file} does not have exactly 5 fields."
                    )

                class_id, x_center, y_center, width, height = fields
                self._validate_or_fix_annotation(
                    class_id=class_id,
                    x_center=x_center,
                    y_center=y_center,
                    width=width,
                    height=height,
                    line_num=line_num,
                    annotation_file=annotation_file,
                )
            except ValueError as e:
                raise ValueError(
                    f"Error in line {line_num} of {annotation_file}: {str(e)}"
                )

    def _validate_or_fix_annotation(
        self,
        class_id: float,
        x_center: float,
        y_center: float,
        width: float,
        height: float,
        line_num: int,
        annotation_file: str,
    ):
        """
        Validates or fixes a single YOLO annotation.

        This method checks whether each annotation field (class ID, center coordinates, width, and height)
        is within the valid range. If an issue is found and `fix_annotation` is enabled, it corrects the value.

        Args:
            class_id (float): The class ID.
            x_center (float): The x center coordinate (normalized).
            y_center (float): The y center coordinate (normalized).
            width (float): The width (normalized).
            height (float): The height (normalized).
            line_num (int): The line number in the annotation file.
            annotation_file (str): The name of the annotation file.

        Returns:
            None: If the annotation is valid, or if fixed, the annotation file is updated.
        """
        if class_id < 0 or class_id >= len(self.dataset_context.labelmap):
            self.error_count["class_id"] += 1
            if self.fix_annotation:
                class_id = max(
                    0, min(len(self.dataset_context.labelmap) - 1, int(class_id))
                )

        if not (0 <= x_center <= 1):
            self.error_count["x_center"] += 1
            if self.fix_annotation:
                x_center = max(0, min(1, x_center))

        if not (0 <= y_center <= 1):
            self.error_count["y_center"] += 1
            if self.fix_annotation:
                y_center = max(0, min(1, y_center))

        if not (0 < width <= 1):
            self.error_count["width"] += 1
            if self.fix_annotation:
                width = max(0.01, min(1, width))

        if not (0 < height <= 1):
            self.error_count["height"] += 1
            if self.fix_annotation:
                height = max(0.01, min(1, height))

        if self.fix_annotation:
            new_line = f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            self._update_annotation_file(
                annotation_file=annotation_file, line_num=line_num, new_line=new_line
            )

    def _update_annotation_file(
        self, annotation_file: str, line_num: int, new_line: str
    ):
        """
        Updates a specific line in the annotation file.

        This method replaces the old line with the corrected annotation line in the file.

        Args:
            annotation_file (str): The name of the annotation file to update.
            line_num (int): The line number to update.
            new_line (str): The updated line content.
        """
        if not self.dataset_context.annotations_dir:
            print(
                f"Annotations directory is missing for dataset {self.dataset_context.dataset_name}, skipping update."
            )
            return
        annotation_path = os.path.join(
            self.dataset_context.annotations_dir, annotation_file
        )
        with open(annotation_path, "r") as file:
            lines = file.readlines()
        lines[line_num - 1] = new_line
        with open(annotation_path, "w") as file:
            file.writelines(lines)

    def _report_errors(self):
        """
        Reports validation errors found in the dataset annotations.

        This method outputs a summary of the validation errors, including the number of issues found
        for each field (class ID, center coordinates, width, and height). If `fix_annotation` is enabled,
        the issues are automatically corrected.
        """
        print(f"⚠️ Found {sum(self.error_count.values())} YOLO annotation issues:")
        for error_type, count in self.error_count.items():
            print(f" - {error_type}: {count} issues")

        if self.fix_annotation:
            print("🔧 Fixing these issues automatically...")
        else:
            raise ValueError(
                "YOLO annotation issues detected. Set 'fix_annotation' to True to automatically fix them."
            )
