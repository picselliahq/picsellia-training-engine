import os
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

    def preprocess_dataset(self, dataset_context: TDatasetContext) -> List[str]:
        """
        Prépare et retourne la liste des chemins d'images à partir du `dataset_context`.
        """
        image_paths = []
        for category_name in os.listdir(dataset_context.image_dir):
            category_dir = os.path.join(dataset_context.image_dir, category_name)
            image_paths += [
                os.path.join(category_dir, image_name)
                for image_name in os.listdir(category_dir)
            ]
        return image_paths

    def prepare_batches(
        self, image_paths: List[str], batch_size: int
    ) -> List[List[str]]:
        """
        Divise la liste de chemins d'images en lots (batches) de taille `batch_size`.
        """
        batches = [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]
        return batches

    def run_inference_on_batches(
        self,
        batches: List[List[str]],
    ) -> List[Results]:
        """
        Exécute l'inférence sur chaque lot d'images et retourne toutes les prédictions.
        """
        all_predictions = []

        for batch_image_paths in batches:
            batch_predictions = self._run_inference(batch_image_paths)
            all_predictions.append(batch_predictions)
        return all_predictions

    def _run_inference(self, image_paths: List[str]) -> Results:
        return self.model_context.loaded_model(image_paths)

    def post_process_batches(
        self,
        batches: List[List[str]],
        results: List[Results],
        dataset_context: TDatasetContext,
    ) -> PredictionClassificationResult:
        """
        Exécute le post-traitement sur chaque lot de prédictions et retourne un seul PredictionClassificationResult.
        """
        all_image_paths = []
        all_classes = []
        all_confidences = []

        # Traite chaque lot (batch)
        for predictions, image_paths in zip(results, batches):
            classification_prediction = self._post_process(
                image_paths=image_paths,
                predictions=predictions,
                dataset_context=dataset_context,
            )

            # Concatène les chemins d'images, les labels et les confidences de chaque batch
            all_image_paths.extend(classification_prediction.image_paths)
            all_classes.extend(classification_prediction.classes)
            all_confidences.extend(classification_prediction.confidences)

        # Crée un unique PredictionClassificationResult en fusionnant tous les résultats
        final_result = PredictionClassificationResult(
            image_paths=all_image_paths,
            classes=all_classes,
            confidences=all_confidences,
        )

        return final_result

    def _post_process(
        self, image_paths, predictions: Results, dataset_context: TDatasetContext
    ) -> PredictionClassificationResult:
        """
        Post-traitement pour un lot d'images et leurs prédictions.
        """
        classes = []
        confidences = []

        for prediction in predictions:
            classes.append(
                [
                    self.get_picsellia_label(
                        prediction.names[int(prediction.probs.top1)], dataset_context
                    )
                ]
            )
            confidences.append(
                [
                    self.get_picsellia_confidence(
                        float(prediction.probs.top1conf.numpy())
                    )
                ]
            )

        return PredictionClassificationResult(
            image_paths=image_paths, classes=classes, confidences=confidences
        )
