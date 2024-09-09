from typing import Optional

from picsellia.types.enums import InferenceType

from src.models.steps.processing.dataset_version_creation.tiler_processing.base_tiler_processing import (
    BaseTilerProcessing,
    TileMode,
)
from src.models.steps.processing.dataset_version_creation.tiler_processing.classification_tiler_processing import (
    ClassificationTilerProcessing,
)
from src.models.steps.processing.dataset_version_creation.tiler_processing.object_detection_tiler_processing import (
    ObjectDetectionTilerProcessing,
)
from src.models.steps.processing.dataset_version_creation.tiler_processing.segmentation_tiler_processing import (
    SegmentationTilerProcessing,
)


class TilerProcessingFactory:
    """Factory class to create the appropriate TilerProcessing instance."""

    @staticmethod
    def create_tiler_processing(
        dataset_type: InferenceType,
        tile_height: int,
        tile_width: int,
        overlap_height_ratio: float,
        overlap_width_ratio: float,
        min_annotation_area_ratio: Optional[float],
        min_annotation_width: Optional[int],
        min_annotation_height: Optional[int],
        tilling_mode: TileMode = TileMode.CONSTANT,
        constant_value: int = 114,
    ) -> BaseTilerProcessing:
        """Create and return the appropriate TilerProcessing instance based on the dataset type."""

        if dataset_type == InferenceType.OBJECT_DETECTION:
            return ObjectDetectionTilerProcessing(
                tile_height=tile_height,
                tile_width=tile_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_annotation_area_ratio=min_annotation_area_ratio,
                min_annotation_width=min_annotation_width,
                min_annotation_height=min_annotation_height,
                tilling_mode=tilling_mode,
                constant_value=constant_value,
            )

        elif dataset_type == InferenceType.SEGMENTATION:
            return SegmentationTilerProcessing(
                tile_height=tile_height,
                tile_width=tile_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_annotation_area_ratio=min_annotation_area_ratio,
                min_annotation_width=min_annotation_width,
                min_annotation_height=min_annotation_height,
                tilling_mode=tilling_mode,
                constant_value=constant_value,
            )

        elif dataset_type == InferenceType.CLASSIFICATION:
            return ClassificationTilerProcessing(
                tile_height=tile_height,
                tile_width=tile_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_annotation_area_ratio=min_annotation_area_ratio,
                min_annotation_width=min_annotation_width,
                min_annotation_height=min_annotation_height,
                tilling_mode=tilling_mode,
                constant_value=constant_value,
            )
        else:
            raise ValueError(
                f"The provided dataset type {dataset_type} is not supported yet for tiling."
            )
