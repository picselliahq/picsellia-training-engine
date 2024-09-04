import os
from abc import ABC
from typing import TypeVar, Generic, List

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.common.model_context import ModelContext
from src.models.model.common.picsellia_prediction import (
    PicselliaClassificationPrediction,
    PredictionClassificationResult,
    PicselliaRectanglePrediction,
    PredictionRectangleResult,
    PicselliaLabel,
    PicselliaConfidence,
    PicselliaRectangle,
)

TModelContext = TypeVar("TModelContext", bound=ModelContext)


class ModelContextInference(ABC, Generic[TModelContext]):
    def __init__(self, model_context: TModelContext):
        self.model_context: TModelContext = model_context

    def get_picsellia_label(
        self, category_name: str, dataset_context: TDatasetContext
    ) -> PicselliaLabel:
        return PicselliaLabel(
            dataset_context.dataset_version.get_or_create_label(category_name)
        )

    def get_picsellia_confidence(self, confidence: float) -> PicselliaConfidence:
        return PicselliaConfidence(confidence)


class ClassificationModelContextInference(ModelContextInference[TModelContext]):
    def __init__(self, model_context: TModelContext):
        super().__init__(model_context)

    def get_picsellia_predictions(
        self,
        dataset_context: TDatasetContext,
        prediction_result: PredictionClassificationResult,
    ) -> List[PicselliaClassificationPrediction]:
        predictions = []
        for i in range(len(prediction_result)):
            result = prediction_result[i]
            asset_id = os.path.basename(result["image_path"]).split(".")[0]
            asset = dataset_context.dataset_version.find_all_assets(ids=[asset_id])[0]
            prediction = PicselliaClassificationPrediction(
                asset,
                result["classes"],
                result["confidences"],
            )
            predictions.append(prediction)
        return predictions


class RectangleModelContextInference(ModelContextInference[TModelContext]):
    def __init__(self, model_context: TModelContext):
        super().__init__(model_context)

    def get_picsellia_predictions(
        self,
        dataset_context: TDatasetContext,
        prediction_result: PredictionRectangleResult,
    ) -> List[PicselliaRectanglePrediction]:
        predictions = []
        for i in range(len(prediction_result)):
            result = prediction_result[i]
            asset_id = os.path.basename(result["image_path"]).split(".")[0]
            asset = dataset_context.dataset_version.find_all_assets(ids=[asset_id])[0]
            prediction = PicselliaRectanglePrediction(
                asset,
                result["boxes"],
                result["labels"],
                result["confidences"],
            )
            predictions.append(prediction)
        return predictions

    def get_picsellia_rectangle(
        self, x: int, y: int, w: int, h: int
    ) -> PicselliaRectangle:
        return PicselliaRectangle(x=x, y=y, w=w, h=h)
