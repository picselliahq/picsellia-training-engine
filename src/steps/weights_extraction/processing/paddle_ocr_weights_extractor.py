from src import step, Pipeline
from src.models.contexts.processing.picsellia_processing_context import PicselliaProcessingContext
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.steps.weights_extraction.processing.paddle_ocr_model_collection_extractor import (
    PaddleOCRModelCollectionExtractor,
)


@step
def paddle_ocr_weights_extractor() -> PaddleOCRModelCollection:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    model_collection_extractor = PaddleOCRModelCollectionExtractor(
        model_version=context.model_version
    )
    model_collection = model_collection_extractor.get_model_collection()
    model_collection.download_weights()
    return model_collection
