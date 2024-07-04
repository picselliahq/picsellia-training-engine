from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection
from src.models.steps.weights_extraction.training.paddle_ocr_model_collection_extractor import (
    PaddleOCRModelCollectionExtractor,
)


@step
def paddle_ocr_weights_extractor() -> PaddleOCRModelCollection:
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_collection_extractor = PaddleOCRModelCollectionExtractor(
        experiment=context.experiment
    )
    model_collection = model_collection_extractor.get_model_collection()
    model_collection.download_weights()
    return model_collection
