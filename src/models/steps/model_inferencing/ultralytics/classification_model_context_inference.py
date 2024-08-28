from typing import List

from ultralytics.engine.results import Results

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.common.model_context import ModelContext
from src.models.model.common.picsellia_prediction import (
    PredictionClassificationResult,
)
from src.models.steps.model_inferencing.model_context_inference import (
    ClassificationModelContextInference,
)


class UltralyticsClassificationModelContextInference(
    ClassificationModelContextInference[ModelContext]
):
    def __init__(self, model_context: ModelContext):
        super().__init__(model_context)

    def predict_images(
        self, image_paths: List[str], dataset_context: TDatasetContext
    ) -> PredictionClassificationResult:
        predictions = self._run_inference(image_paths)
        prediction_classification_result = self._parse_predictions(
            image_paths, predictions, dataset_context
        )
        return prediction_classification_result

    def _run_inference(self, image_paths: List[str]) -> Results:
        return self.model_context.loaded_model(image_paths)

    def _parse_predictions(
        self, image_paths, predictions: Results, dataset_context: TDatasetContext
    ) -> PredictionClassificationResult:
        labels_per_image = []
        confidences_per_image = []
        for prediction in predictions:
            labels_per_image.append(
                [
                    self.get_picsellia_label(
                        prediction.names[int(prediction.probs.top1)], dataset_context
                    )
                ]
            )
            confidences_per_image.append(
                [
                    self.get_picsellia_confidence(
                        float(prediction.probs.top1conf.numpy())
                    )
                ]
            )
        print(f"labels_per_image: {labels_per_image}")
        print(f"confidences_per_image: {confidences_per_image}")
        return PredictionClassificationResult(
            image_paths, labels_per_image, confidences_per_image
        )
