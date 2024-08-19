from picsellia.types.enums import InferenceType

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.parameters.processing.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from src.models.steps.data_validation.common.not_configured_dataset_context_validator import (
    NotConfiguredDatasetContextValidator,
)
from src.models.steps.data_validation.common.object_detection_dataset_context_validator import (
    ObjectDetectionDatasetContextValidator,
)
from src.models.steps.data_validation.common.segmentation_dataset_context_validator import (
    SegmentationDatasetContextValidator,
)
from src.models.steps.data_validation.processing.processing_tiler_data_validator import (
    ProcessingTilerDataValidator,
)


@step
def tiler_data_validator(
    dataset_context: DatasetContext,
) -> None:
    context: PicselliaProcessingContext[
        ProcessingTilerParameters
    ] = Pipeline.get_active_context()

    match dataset_context.dataset_version.type:
        case InferenceType.NOT_CONFIGURED:
            not_configured_dataset_validator = NotConfiguredDatasetContextValidator(
                dataset_context=dataset_context
            )
            not_configured_dataset_validator.validate()

        case InferenceType.SEGMENTATION:
            object_detection_dataset_validator = ObjectDetectionDatasetContextValidator(
                dataset_context=dataset_context
            )
            segmentation_dataset_validator = SegmentationDatasetContextValidator(
                dataset_context=dataset_context
            )

            # SAHI requires both the bounding boxes and the polygons to be valid
            object_detection_dataset_validator.validate()
            segmentation_dataset_validator.validate()

        case InferenceType.OBJECT_DETECTION:
            object_detection_dataset_validator = ObjectDetectionDatasetContextValidator(
                dataset_context=dataset_context
            )
            object_detection_dataset_validator.validate()

        case _:
            raise ValueError(
                f"Dataset type {dataset_context.dataset_version.type} is not supported."
            )

    processing_validator = ProcessingTilerDataValidator(
        client=context.client,
        tile_height=context.processing_parameters.tile_height,
        tile_width=context.processing_parameters.tile_width,
        datalake=context.processing_parameters.datalake,
    )
    processing_validator.validate()
