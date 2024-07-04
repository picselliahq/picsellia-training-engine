import os

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.parameters.processing.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from src.models.steps.processing.dataset_version_creation.bounding_box_cropper_processing import (
    BoundingBoxCropperProcessing,
)


@step
def bounding_box_cropper_processing(dataset_context: DatasetContext):
    context: PicselliaProcessingContext[
        ProcessingBoundingBoxCropperParameters
    ] = Pipeline.get_active_context()

    processor = BoundingBoxCropperProcessing(
        client=context.client,
        datalake=context.client.get_datalake(context.processing_parameters.datalake),
        input_dataset_context=dataset_context,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        output_dataset_version=context.output_dataset_version,
        destination_path=os.path.join(os.getcwd(), str(context.job_id)),
    )
    processor.process()
