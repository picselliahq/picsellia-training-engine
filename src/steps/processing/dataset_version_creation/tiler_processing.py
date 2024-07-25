from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)
from src.models.parameters.processing.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from src.models.steps.processing.dataset_version_creation.tiler_processing import (
    TilerProcessing,
)


@step
def tiler_processing(
    dataset_collection: ProcessingDatasetCollection,
) -> DatasetContext:
    context: PicselliaProcessingContext[
        ProcessingTilerParameters
    ] = Pipeline.get_active_context()

    processor = TilerProcessing(
        dataset_collection=dataset_collection,
        tile_height=context.processing_parameters.tile_height,
        tile_width=context.processing_parameters.tile_width,
        overlap_height_ratio=context.processing_parameters.overlap_height_ratio,
        overlap_width_ratio=context.processing_parameters.overlap_width_ratio,
        min_area_ratio=context.processing_parameters.min_area_ratio,
    )
    dataset_collection = processor.process()
    return dataset_collection.output