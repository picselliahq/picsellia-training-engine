from typing import Union
from picsellia_annotations.coco import Annotation
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class ObjectDetectionDatasetContextValidator(DatasetContextValidator):
    def __init__(self, dataset_context, fix_annotation=True):
        super().__init__(dataset_context)
        self.fix_annotation = fix_annotation
        self.error_count = {
            "top_left_x": 0,
            "top_left_y": 0,
            "bottom_right_x": 0,
            "bottom_right_y": 0,
        }

    def validate(self):
        """
        Validate the object detection dataset context.
        An object detection dataset context must have at least 1 class and at least 1 image with bounding boxes.

        Raises:
            ValueError: If the object detection dataset context is not valid and fix_annotation is False.
        """
        super().validate()
        self._validate_labelmap()
        self._validate_at_least_one_image_with_bounding_boxes()
        self._validate_bounding_boxes_coordinates()

        if any(self.error_count.values()):
            self._report_errors()

        return self.dataset_context

    def _validate_labelmap(self):
        """
        Validate that the labelmap for the dataset context is valid.
        An object detection labelmap must have at least 1 class.

        Raises:
            ValueError: If the labelmap for the dataset context is not valid.
        """
        if len(self.dataset_context.labelmap) < 1:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"An object detection labelmap must have at least 1 class."
            )

    def _validate_at_least_one_image_with_bounding_boxes(self):
        """
        Validate that the dataset context has at least 1 image with bounding boxes.
        If no images have bounding boxes, the dataset is considered invalid.

        Raises:
            ValueError: If the dataset context has no images with bounding boxes.
        """
        if not self.dataset_context.coco_file.annotations:
            raise ValueError(
                f"Dataset {self.dataset_context.dataset_name} must have at least 1 image with bounding boxes."
            )

    def _validate_bounding_boxes_coordinates(self):
        """
        Validate the bounding box coordinates for all annotations in the dataset context.
        Bounding box coordinates are greater than or equal to 0, and the bottom-right coordinates
        must be greater than the top-left coordinates.

        Raises:
            ValueError: If the bounding box coordinates for any annotation are not valid and fix_annotation is False.
        """
        for num_annotation, annotation in enumerate(
            self.dataset_context.coco_file.annotations
        ):
            modified_annotation = self._fix_or_count_errors(annotation)
            if modified_annotation:
                self.dataset_context.coco_file.annotations[
                    num_annotation
                ] = modified_annotation

    def _fix_or_count_errors(self, annotation: Annotation) -> Union[Annotation, None]:
        """
        Fix or count errors in the bounding box coordinates for a given annotation.
        Args:
            annotation (Annotation): The annotation to validate and potentially fix.
        """
        x, y, width, height = annotation.bbox
        top_left_x, top_left_y = x, y
        bottom_right_x = x + width
        bottom_right_y = y + height
        image = self._get_image_by_id(annotation.image_id)

        (
            top_left_x,
            top_left_y,
            bottom_right_x,
            bottom_right_y,
        ) = self._correct_coordinates_if_needed(
            top_left_x, top_left_y, bottom_right_x, bottom_right_y, image
        )

        # If fix_annotation is True, update the annotation directly
        if self.fix_annotation:
            new_width = bottom_right_x - top_left_x
            new_height = bottom_right_y - top_left_y
            annotation.bbox = [top_left_x, top_left_y, new_width, new_height]
            return annotation
        return None

    def _correct_coordinates_if_needed(
        self, top_left_x, top_left_y, bottom_right_x, bottom_right_y, image
    ):
        """
        Check and correct the bounding box coordinates if they are invalid.
        Args:
            top_left_x, top_left_y, bottom_right_x, bottom_right_y: Coordinates of the bounding box.
            image: The image object related to the annotation.
        Returns:
            Corrected top_left_x, top_left_y, bottom_right_x, bottom_right_y values.
        """
        if top_left_x < 0:
            self.error_count["top_left_x"] += 1
            if self.fix_annotation:
                top_left_x = 0

        if top_left_y < 0:
            self.error_count["top_left_y"] += 1
            if self.fix_annotation:
                top_left_y = 0

        if bottom_right_x > image.width or bottom_right_x <= top_left_x:
            self.error_count["bottom_right_x"] += 1
            if self.fix_annotation:
                bottom_right_x = max(top_left_x + 1, image.width)

        if bottom_right_y > image.height or bottom_right_y <= top_left_y:
            self.error_count["bottom_right_y"] += 1
            if self.fix_annotation:
                bottom_right_y = max(top_left_y + 1, image.height)

        return top_left_x, top_left_y, bottom_right_x, bottom_right_y

    def _get_image_by_id(self, image_id):
        """
        Get the image object by its ID.
        Args:
            image_id: The ID of the image to retrieve.
        Returns:
            The image object associated with the given ID.
        """
        return next(
            image
            for image in self.dataset_context.coco_file.images
            if image.id == image_id
        )

    def _report_errors(self):
        """
        Report the errors found during validation.
        """
        print(f"⚠️ Found {sum(self.error_count.values())} bounding box issues:")
        for error_type, count in self.error_count.items():
            print(f" - {error_type}: {count} issues")

        if self.fix_annotation:
            print("🔧 Fixing these issues automatically...")
        else:
            raise ValueError(
                "Bounding box issues detected. Set 'fix_annotation' to True to automatically fix them."
            )
