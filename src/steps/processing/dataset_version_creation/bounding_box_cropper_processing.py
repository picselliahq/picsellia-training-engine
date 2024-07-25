from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)
from src.models.parameters.processing.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from src.models.steps.processing.dataset_version_creation.bounding_box_cropper_processing import (
    BoundingBoxCropperProcessing,
)


@step
def bounding_box_cropper_processing(
    dataset_collection: ProcessingDatasetCollection,
) -> DatasetContext:
    context: PicselliaProcessingContext[
        ProcessingBoundingBoxCropperParameters
    ] = Pipeline.get_active_context()

    processor = BoundingBoxCropperProcessing(
        dataset_collection=dataset_collection,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
    )
    dataset_collection = processor.process()
    return dataset_collection.output
