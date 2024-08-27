from src.models.model.common.model_collection import ModelCollection
from src.models.model.common.model_context import ModelContext


class PaddleOCRModelCollection(ModelCollection):
    """
    A specialized collection for Paddle OCR models that contains two models:
    one for bounding box detection and one for text recognition.

    Attributes:
        bbox_model (ModelContext): The model context for the bounding box detection model.
        text_model (ModelContext): The model context for the text recognition model.
    """

    def __init__(self, bbox_model: ModelContext, text_model: ModelContext):
        """
        Initializes a new PaddleOCRModelCollection with specified contexts for bounding box and text recognition models.

        Args:
            bbox_model (ModelContext): The model context for the bounding box detection.
            text_model (ModelContext): The model context for the text recognition.
        """
        self.bbox_model = bbox_model
        self.text_model = text_model

    def download_weights(self):
        """
        Downloads the weights for both bounding box and text recognition models.
        """
        self.bbox_model.download_weights()
        self.text_model.download_weights()
