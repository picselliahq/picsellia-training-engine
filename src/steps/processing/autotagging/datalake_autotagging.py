from typing import Union

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.processing.datalake_collection import DatalakeCollection
from src.models.dataset.processing.datalake_context import DatalakeContext
from src.models.parameters.processing.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from src.models.steps.processing.autotagging.datalake_autotagging_processing import DatalakeAutotaggingProcessing



@step
def datalake_autotagging_processing(
    datalake: Union[DatalakeContext, DatalakeCollection],
) -> DatasetContext:
    context: PicselliaProcessingContext[
        ProcessingBoundingBoxCropperParameters
    ] = Pipeline.get_active_context()

    processor = DatalakeAutotaggingProcessing(
        datalake=datalake,
        tags_list=context.processing_parameters.tags_list,
    )
    dataset_collection = processor.process()
    return dataset_collection.output
