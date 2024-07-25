from src import Pipeline, step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection

from src.models.parameters.processing.processing_paddleocr_parameters import (
    ProcessingPaddleOcrParameters,
)
from src.models.steps.processing.pre_annotation.paddleocr_processing import (
    PaddleOcrProcessing,
)


@step
def paddleocr_processing(
    dataset_context: DatasetContext, model_collection: PaddleOCRModelCollection
):
    context: PicselliaProcessingContext[
        ProcessingPaddleOcrParameters
    ] = Pipeline.get_active_context()

    processor = PaddleOcrProcessing(
        client=context.client,
        input_dataset_context=dataset_context,
        model_collection=model_collection,
    )
    processor.process()
