from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection

from src.models.steps.weights_extraction.training.training_model_context_extractor import (
    TrainingModelContextExtractor,
)


class PaddleOCRModelCollectionExtractor(TrainingModelContextExtractor):
    def get_model_collection(self) -> PaddleOCRModelCollection:
        return PaddleOCRModelCollection(
            bbox_model=self.get_model_context(prefixed_model_name="bbox"),
            text_model=self.get_model_context(prefixed_model_name="text"),
        )
