import os

from src import Pipeline, step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.parameters.processing.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from src.models.parameters.processing.processing_easyorc_parameters import ProcessingEasyOcrParameters
from src.models.steps.processing.dataset_version_creation.bounding_box_cropper_processing import (
    BoundingBoxCropperProcessing,
)
from src.models.steps.processing.pre_annotation.easyocr_processing import EasyOcrProcessing


@step
def easyocr_processing(dataset_context: DatasetContext):
    context: PicselliaProcessingContext[
        ProcessingEasyOcrParameters
    ] = Pipeline.get_active_context()

    processor = EasyOcrProcessing(
        client=context.client,
        input_dataset_context=dataset_context,
        language=context.processing_parameters.language,
    )
    processor.process()
