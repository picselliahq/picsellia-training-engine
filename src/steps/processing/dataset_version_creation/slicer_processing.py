from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.common.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)
from src.models.parameters.processing.processing_slicer_parameters import (
    ProcessingSlicerParameters,
)
from src.models.steps.processing.dataset_version_creation.slicer_processing import (
    SlicerProcessing,
)


@step
def slicer_processing(
    dataset_collection: ProcessingDatasetCollection,
) -> ProcessingDatasetCollection:
    context: PicselliaProcessingContext[
        ProcessingSlicerParameters
    ] = Pipeline.get_active_context()

    processor = SlicerProcessing(
        dataset_collection=dataset_collection,
        slice_height=context.processing_parameters.slice_height,
        slice_width=context.processing_parameters.slice_width,
        overlap_height_ratio=context.processing_parameters.overlap_height_ratio,
        overlap_width_ratio=context.processing_parameters.overlap_width_ratio,
        min_area_ratio=context.processing_parameters.min_area_ratio,
    )
    dataset_collection = processor.process()
    return dataset_collection
