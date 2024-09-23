from abc import ABC
from typing import Generic

from src.models.dataset.common.dataset_context import TDatasetContext
from src.models.model.common.model_context import TModelContext
from src.models.model.common.picsellia_prediction import (
    PicselliaLabel,
    PicselliaConfidence,
    PicselliaRectangle,
)


class ModelContextPredictor(ABC, Generic[TModelContext]):
    def __init__(self, model_context: TModelContext):
        """
        Initializes the base class for performing inference using a model context.

        Args:
            model_context (TModelContext): The context containing the loaded model and configurations.
        """
        self.model_context: TModelContext = model_context

        if not hasattr(self.model_context, "loaded_model"):
            raise ValueError(
                "The model context does not have a loaded model attribute."
            )

    def get_picsellia_label(
        self, category_name: str, dataset_context: TDatasetContext
    ) -> PicselliaLabel:
        """
        Retrieves or creates a label for a given category name within the dataset context.

        Args:
            category_name (str): The name of the category to retrieve the label for.
            dataset_context (TDatasetContext): The dataset context containing the label information.

        Returns:
            PicselliaLabel: The corresponding Picsellia label for the given category.
        """
        return PicselliaLabel(
            dataset_context.dataset_version.get_or_create_label(category_name)
        )

    def get_picsellia_confidence(self, confidence: float) -> PicselliaConfidence:
        """
        Converts a confidence score into a PicselliaConfidence object.

        Args:
            confidence (float): The confidence score for the prediction.

        Returns:
            PicselliaConfidence: The confidence score wrapped in a PicselliaConfidence object.
        """
        return PicselliaConfidence(confidence)

    def get_picsellia_rectangle(
        self, x: int, y: int, w: int, h: int
    ) -> PicselliaRectangle:
        """
        Creates a PicselliaRectangle object representing a bounding box.

        Args:
            x (int): The x-coordinate of the top-left corner of the rectangle.
            y (int): The y-coordinate of the top-left corner of the rectangle.
            w (int): The width of the rectangle.
            h (int): The height of the rectangle.

        Returns:
            PicselliaRectangle: The rectangle object with the specified dimensions.
        """
        return PicselliaRectangle(x=x, y=y, w=w, h=h)
