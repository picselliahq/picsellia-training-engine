import os

from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.models.parameters.processing.processing_slicer_parameters import (
    ProcessingSlicerParameters,
)
from src.models.steps.processing.dataset_version_creation.slicer_processing import (
    SlicerProcessing,
)


@step
def slicer_processing(dataset_context: DatasetContext):
    context: PicselliaProcessingContext[
        ProcessingSlicerParameters
    ] = Pipeline.get_active_context()

    processor = SlicerProcessing(
        client=context.client,
        input_dataset_context=dataset_context,
        slice_height=context.processing_parameters.slice_height,
        slice_width=context.processing_parameters.slice_width,
        overlap_height_ratio=context.processing_parameters.overlap_height_ratio,
        overlap_width_ratio=context.processing_parameters.overlap_width_ratio,
        min_area_ratio=context.processing_parameters.min_area_ratio,
        output_dataset_version=context.output_dataset_version,
        datalake=context.processing_parameters.datalake,
        destination_path=os.path.join(
            os.getcwd(), str(context.job_id), str("processed_dataset")
        ),
    )
    processor.process()
