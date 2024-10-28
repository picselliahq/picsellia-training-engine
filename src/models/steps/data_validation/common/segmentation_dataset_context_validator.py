from typing import Union
from picsellia_annotations.coco import Annotation

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class SegmentationDatasetContextValidator(DatasetContextValidator):
    def __init__(self, dataset_context: DatasetContext, fix_annotation=False):
        super().__init__(dataset_context=dataset_context, fix_annotation=fix_annotation)
        self.error_count = {
            "x_coord": 0,
            "y_coord": 0,
        }

    def validate(self):
        """
        Validate the segmentation dataset context.
        A segmentation dataset context must have at least 1 class and at least 1 image with polygons.

        Raises:
            ValueError: If the segmentation dataset context is not valid and fix_annotation is False.
        """
        super().validate()
        self._validate_labelmap()
        self._validate_at_least_one_image_with_polygons()
        self._validate_polygons_coordinates()

        if any(self.error_count.values()):
            self._report_errors()

        return self.dataset_context

    def _validate_labelmap(self):
        """
        Validate that the labelmap for the dataset context is valid.
        A segmentation labelmap must have at least 1 class.

        Raises:
            ValueError: If the labelmap for the dataset context is not valid.
        """
        if len(self.dataset_context.labelmap) < 1:
            raise ValueError(
                f"Labelmap for dataset {self.dataset_context.dataset_name} is not valid. "
                f"A segmentation labelmap must have at least 1 class."
            )

    def _validate_at_least_one_image_with_polygons(self):
        """
        Validate that the dataset context has at least 1 image with polygons.

        Raises:
            ValueError: If the dataset context has no images with polygons.
        """
        if not self.dataset_context.coco_file:
            raise ValueError(
                f"Coco file for dataset {self.dataset_context.dataset_name} is missing."
            )
        if not self.dataset_context.coco_file.annotations:
            raise ValueError(
                f"Dataset {self.dataset_context.dataset_name} must have at least 1 image with polygons."
            )

    def _validate_polygons_coordinates(self):
        """
        Validate the polygon coordinates for all annotations in the dataset context.
        Polygon coordinates are greater than or equal to 0 and are within the image dimensions.

        Raises:
            ValueError: If the polygon coordinates for any annotation are not valid and fix_annotation is False.
        """
        if not self.dataset_context.coco_file:
            raise (ValueError("Coco file is missing."))
        elif not self.dataset_context.coco_file.annotations:
            raise (ValueError("No annotations found in the coco file."))
        for num_annotation, annotation in enumerate(
            self.dataset_context.coco_file.annotations
        ):
            modified_annotation = self._fix_or_count_polygon_errors(annotation)
            if modified_annotation:
                self.dataset_context.coco_file.annotations[
                    num_annotation
                ] = modified_annotation

    def _fix_or_count_polygon_errors(
        self, annotation: Annotation
    ) -> Union[Annotation, None]:
        """
        Fix or count errors in the polygon coordinates for a given annotation.
        Args:
            annotation (Annotation): The annotation to validate and potentially fix.
        Returns:
            Optional[Annotation]: The modified annotation if fix_annotation is True, otherwise None.
        """
        image = self._get_image_by_id(annotation.image_id)
        new_segmentation = []

        for segmentation in annotation.segmentation:
            corrected_segmentation = self._correct_polygon_coordinates_if_needed(
                segmentation, image
            )
            new_segmentation.append(corrected_segmentation)

        if self.fix_annotation:
            annotation.segmentation = new_segmentation
            return annotation
        return None

    def _correct_polygon_coordinates_if_needed(self, segmentation, image):
        """
        Check and correct the polygon coordinates if they are invalid.
        Args:
            segmentation: The list of coordinates representing the polygon.
            image: The image object related to the annotation.
        Returns:
            Corrected list of coordinates for the polygon.
        """
        corrected_segmentation = []
        for i, coordinate in enumerate(segmentation):
            if i % 2 == 0:  # x coordinate
                if coordinate < 0:
                    self.error_count["x_coord"] += 1
                    coordinate = 0 if self.fix_annotation else coordinate
                elif coordinate > image.width:
                    self.error_count["x_coord"] += 1
                    coordinate = image.width if self.fix_annotation else coordinate
            else:  # y coordinate
                if coordinate < 0:
                    self.error_count["y_coord"] += 1
                    coordinate = 0 if self.fix_annotation else coordinate
                elif coordinate > image.height:
                    self.error_count["y_coord"] += 1
                    coordinate = image.height if self.fix_annotation else coordinate

            corrected_segmentation.append(coordinate)

        return corrected_segmentation

    def _get_image_by_id(self, image_id):
        """
        Get the image object by its ID.
        Args:
            image_id: The ID of the image to retrieve.
        Returns:
            The image object associated with the given ID.
        """
        if not self.dataset_context.coco_file:
            raise ValueError("Coco file is missing.")
        if not self.dataset_context.coco_file.images:
            raise ValueError("No images found in the coco file.")
        return next(
            image
            for image in self.dataset_context.coco_file.images
            if image.id == image_id
        )

    def _report_errors(self):
        """
        Report the errors found during validation.
        """
        print(f"⚠️ Found {sum(self.error_count.values())} polygon coordinate issues:")
        for error_type, count in self.error_count.items():
            print(f" - {error_type}: {count} issues")

        if self.fix_annotation:
            print("🔧 Fixing these issues automatically...")
        else:
            raise ValueError(
                "Polygon coordinate issues detected. Set 'fix_annotation' to True to automatically fix them."
            )
