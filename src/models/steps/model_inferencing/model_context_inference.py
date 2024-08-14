import os
from abc import abstractmethod
from typing import TypeVar, Generic, List

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.model_context import ModelContext
from src.models.model.picsellia_prediction import (
    PicselliaClassificationPrediction,
    PredictionClassificationResult,
    PicselliaRectanglePrediction,
    PredictionRectangleResult,
    PicselliaLabel,
    PicselliaConfidence,
    PicselliaRectangle,
)
from src.models.steps.model_inferencing.base_model_inference import BaseModelInference

TModelContext = TypeVar("TModelContext", bound=ModelContext)


class ModelContextInference(BaseModelInference, Generic[TModelContext]):
    def __init__(self, model_context: TModelContext):
        self.model_context: TModelContext = model_context

    @abstractmethod
    def predict_on_dataset_context(self, dataset_context: TDatasetContext):
        pass

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

    @abstractmethod
    def predict_images(
        self, image_paths: List[str], dataset_context: TDatasetContext
    ) -> PredictionClassificationResult:
        raise NotImplementedError()

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

    def predict_on_dataset_context(
        self, dataset_context: TDatasetContext, batch_size: int = 32
    ) -> List[PicselliaClassificationPrediction]:
        category_names = os.listdir(dataset_context.image_dir)
        image_paths = [
            os.path.join(dataset_context.image_dir, category_name, image_name)
            for category_name in category_names
            for image_name in os.listdir(
                os.path.join(dataset_context.image_dir, category_name)
            )
        ]

        all_predictions = []
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i : i + batch_size]
            prediction_result = self.predict_images(batch_image_paths, dataset_context)
            batch_predictions = self.get_picsellia_predictions(
                dataset_context, prediction_result
            )
            all_predictions.extend(batch_predictions)

        return all_predictions


class RectangleModelContextInference(ModelContextInference[TModelContext]):
    def __init__(self, model_context: TModelContext):
        super().__init__(model_context)

    @abstractmethod
    def predict_images(
        self, image_paths: List[str], dataset_context: TDatasetContext
    ) -> PredictionRectangleResult:
        raise NotImplementedError()

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

    def predict_on_dataset_context(
        self, dataset_context: TDatasetContext, batch_size: int = 32
    ) -> List[PicselliaRectanglePrediction]:
        image_paths = [
            os.path.join(dataset_context.image_dir, image_name)
            for image_name in os.listdir(dataset_context.image_dir)
        ]

        all_predictions = []
        for i in range(0, len(image_paths), batch_size):
            batch_image_paths = image_paths[i : i + batch_size]
            prediction_result = self.predict_images(batch_image_paths, dataset_context)
            batch_predictions = self.get_picsellia_predictions(
                dataset_context, prediction_result
            )
            all_predictions.extend(batch_predictions)

        return all_predictions

    def get_picsellia_rectangle(
        self, x: int, y: int, w: int, h: int
    ) -> PicselliaRectangle:
        return PicselliaRectangle(x=x, y=y, w=w, h=h)
