import os

from src.models.model.model_context import ModelContext


class PaddleOCRModelCollection:
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

    def update_model_paths(self):
        """
        Update the paths based on potential changes in destination path or new version downloads.
        This can be useful if the model versions are updated and paths need to be reconfigured.
        """
        self.bbox_model.model_weights_path = os.path.join(
            self.bbox_model.destination_path, self.bbox_model.model_name, "weights"
        )
        self.text_model.model_weights_path = os.path.join(
            self.text_model.destination_path, self.text_model.model_name, "weights"
        )

    def __str__(self):
        """
        Provides a string representation of the collection with model details.

        Returns:
            str: A string representation listing both models in the collection.
        """
        return f"PaddleOCRModelCollection(BBox Model: {self.bbox_model.model_name}, Text Model: {self.text_model.model_name})"
