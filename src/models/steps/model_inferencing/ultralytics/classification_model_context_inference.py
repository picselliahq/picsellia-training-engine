from typing import List, Tuple

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.picsellia_prediction import (
    PredictionClassificationResult,
    PicselliaLabel,
    PicselliaConfidence,
)
from src.models.model.ultralytics_model_context import UltralyticsModelContext
from src.models.steps.model_inferencing.model_context_inference import (
    ClassificationModelContextInference,
)


class UltralyticsClassificationModelContextInference(
    ClassificationModelContextInference[UltralyticsModelContext]
):
    def __init__(self, model_context: UltralyticsModelContext):
        super().__init__(model_context)

    def predict_images(
        self, image_paths: List[str], dataset_context: TDatasetContext
    ) -> PredictionClassificationResult:
        predictions = self._run_inference(image_paths)
        labels, confidences = self._parse_predictions(predictions, dataset_context)
        return PredictionClassificationResult(image_paths, labels, confidences)

    def _run_inference(self, image_paths: List[str]):
        return self.model_context.loaded_model(image_paths)

    def _parse_predictions(
        self, predictions, dataset_context: TDatasetContext
    ) -> Tuple[List[List[PicselliaLabel]], List[List[PicselliaConfidence]]]:
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
        return labels_per_image, confidences_per_image
