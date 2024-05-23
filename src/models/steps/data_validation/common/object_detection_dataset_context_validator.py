from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)

from picsellia_annotations.coco import Annotation


class ObjectDetectionDatasetContextValidator(DatasetContextValidator):
    def validate(self):
        """
        Validate the object detection dataset context.
        An object detection dataset context must have at least 1 class and at least 1 image with bounding boxes.

        Raises:
            ValueError: If the object detection dataset context is not valid.
        """
        super().validate()
        self.validate_labelmap()
        self.validate_at_least_one_image_with_bounding_boxes()
        self.validate_bounding_boxes_coordinates()

    def validate_labelmap(self):
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

    def validate_at_least_one_image_with_bounding_boxes(self):
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

    def _validate_bounding_box_coordinates(self, coco_file_annotation: Annotation):
        """
        Validate the bounding box coordinates for an annotation.
        Bounding box coordinates are greater than or equal to 0, have a width and height greater than 0, and are within the image dimensions.
        Args:
            coco_file_annotation (Annotation): The annotation to validate.

        Returns:
            bool: True if the bounding box coordinates are valid, False otherwise.
        """
        x, y, width, height = coco_file_annotation.bbox
        coco_file_image = [
            coco_file_image
            for coco_file_image in self.dataset_context.coco_file.images
            if coco_file_image.id == coco_file_annotation.image_id
        ][0]
        return all(
            [
                x >= 0,
                y >= 0,
                width > 0,
                height > 0,
                x + width <= coco_file_image.width,
                y + height <= coco_file_image.height,
            ]
        )

    def validate_bounding_boxes_coordinates(self):
        """
        Validate the bounding box coordinates for all annotations in the dataset context.
        Bounding box coordinates are greater than or equal to 0, have a width and height greater than 0, and are within the image dimensions.

        Raises:
            ValueError: If the bounding box coordinates for any annotation are not valid.

        """
        coco_file_annotations = self.dataset_context.coco_file.annotations
        for coco_file_annotation in coco_file_annotations:
            if not self._validate_bounding_box_coordinates(coco_file_annotation):
                image = [
                    image
                    for image in self.dataset_context.coco_file.images
                    if image.id == coco_file_annotation.image_id
                ][0]
                raise ValueError(
                    f"Bounding box coordinates for annotation {coco_file_annotation.id} of image {image.file_name} "
                    f"in dataset {self.dataset_context.dataset_name} are not valid. "
                    f"Bounding box coordinates must be integers and have a width and height greater than 0."
                )
