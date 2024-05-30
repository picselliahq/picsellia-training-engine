from picsellia_annotations.coco import Annotation

from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class SegmentationDatasetContextValidator(DatasetContextValidator):
    def validate(self):
        """
        Validate the object detection dataset context.
        An object detection dataset context must have at least 1 class and at least 1 image with bounding boxes.

        Raises:
            ValueError: If the object detection dataset context is not valid.
        """
        super().validate()
        self.validate_labelmap()
        self.validate_at_least_one_image_with_polygons()
        self.validate_polygons_coordinates()

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

    def validate_at_least_one_image_with_polygons(self):
        """
        Validate that the dataset context has at least 1 image with polygons.

        Raises:
            ValueError: If the dataset context has no images with polygons.
        """
        if not self.dataset_context.coco_file.annotations:
            raise ValueError(
                f"Dataset {self.dataset_context.dataset_name} must have at least 1 image with polygons."
            )

    def _validate_polygon_coordinates(self, coco_file_annotation: Annotation):
        """
        Validate the polygon coordinates for a given annotation.
        Polygon coordinates are greater than or equal to 0 and are within the image dimensions.

        Args:
            coco_file_annotation (Annotation): The annotation to validate.

        Raises:
            ValueError: If the polygon coordinates are not valid.
        """
        coco_file_image = [
            coco_file_image
            for coco_file_image in self.dataset_context.coco_file.images
            if coco_file_image.id == coco_file_annotation.image_id
        ][0]
        for segmentation in coco_file_annotation.segmentation:
            for i, coordinate in enumerate(segmentation):
                if i % 2 == 0:
                    if coordinate < 0 or coordinate > coco_file_image.width:
                        raise ValueError(
                            f"Polygon coordinates for annotation {coco_file_annotation.id} "
                            f"of image with filename {coco_file_image.file_name} "
                            f"are not valid. Coordinates must be greater than or equal to 0 "
                            f"and within the image dimensions."
                        )
                else:
                    if coordinate < 0 or coordinate > coco_file_image.height:
                        raise ValueError(
                            f"Polygon coordinates for annotation {coco_file_annotation.id} "
                            f"of image with filename {coco_file_image.file_name} "
                            f"are not valid. Coordinates must be greater than or equal to 0 "
                            f"and within the image dimensions."
                        )

    def validate_polygons_coordinates(self):
        """
        Validate the polygons coordinates for all annotations.
        Polygon coordinates are greater than or equal to 0 and are within the image dimensions.

        Raises:
            ValueError: If the polygons coordinates are not valid.
        """
        for coco_file_annotation in self.dataset_context.coco_file.annotations:
            self._validate_polygon_coordinates(coco_file_annotation)
