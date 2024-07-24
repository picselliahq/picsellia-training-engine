from src.models.model.paddle_ocr_model_collection import PaddleOCRModelCollection

from src.models.steps.weights_extraction.processing.processing_model_context_extractor import \
    ProcessingModelContextExtractor


class PaddleOCRModelCollectionExtractor(ProcessingModelContextExtractor):
    def get_model_collection(self) -> PaddleOCRModelCollection:
        return PaddleOCRModelCollection(
            bbox_model=self.get_model_context(prefix_model_name="bbox"),
            text_model=self.get_model_context(prefix_model_name="text"),
        )
