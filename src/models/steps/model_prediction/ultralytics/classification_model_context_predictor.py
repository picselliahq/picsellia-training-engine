import os
from typing import List

from ultralytics.engine.results import Results

from src.models.dataset.training.training_dataset_collection import TDatasetContext
from src.models.model.common.model_context import ModelContext
from src.models.model.common.picsellia_prediction import (
    PicselliaClassificationPrediction,
)
from src.models.steps.model_prediction.model_context_predictor import (
    ModelContextPredictor,
)


class UltralyticsClassificationModelContextPredictor(
    ModelContextPredictor[ModelContext]
):
    def __init__(self, model_context: ModelContext):
        """
        Initializes the UltralyticsClassificationModelContextInference instance with a provided model context.

        Args:
            model_context (ModelContext): The context containing the loaded model and relevant configurations.
        """
        super().__init__(model_context)

    def pre_process_dataset_context(
        self, dataset_context: TDatasetContext
    ) -> List[str]:
        """
        Prepares and returns a list of image file paths extracted from the provided dataset context.

        Args:
            dataset_context (TDatasetContext): Context containing the dataset and relevant directories.

        Returns:
            List[str]: A list of file paths corresponding to images within the dataset.
        """
        image_paths = []
        for category_name in os.listdir(dataset_context.image_dir):
            category_dir = os.path.join(dataset_context.image_dir, category_name)
            image_paths.extend(
                [
                    os.path.join(category_dir, image_name)
                    for image_name in os.listdir(category_dir)
                ]
            )
        return image_paths

    def prepare_batches(
        self, image_paths: List[str], batch_size: int
    ) -> List[List[str]]:
        """
        Splits the list of image paths into smaller batches of a specified size.

        Args:
            image_paths (List[str]): A list of image file paths.
            batch_size (int): The number of images per batch.

        Returns:
            List[List[str]]: A list of batches, each containing image paths.
        """
        return [
            image_paths[i : i + batch_size]
            for i in range(0, len(image_paths), batch_size)
        ]

    def run_inference_on_batches(
        self,
        image_batches: List[List[str]],
    ) -> List[Results]:
        """
        Performs inference on each batch of images and returns the predictions for all batches.

        Args:
            image_batches (List[List[str]]): A list of image path batches to run inference on.

        Returns:
            List[Results]: A list of prediction results for each batch of images.
        """
        all_batch_results = []

        for batch_paths in image_batches:
            batch_results = self._run_inference(batch_paths)
            all_batch_results.append(batch_results)
        return all_batch_results

    def _run_inference(self, batch_paths: List[str]) -> Results:
        """
        Executes inference on the given list of image paths using the loaded model.

        Args:
            batch_paths (List[str]): A list of image file paths to perform inference on.

        Returns:
            Results: The results of the inference, containing predictions for each image.
        """
        return self.model_context.loaded_model(batch_paths)

    def post_process_batches(
        self,
        image_batches: List[List[str]],
        batch_results: List[Results],
        dataset_context: TDatasetContext,
    ) -> List[PicselliaClassificationPrediction]:
        """
        Post-processes the predictions for each batch and returns a list of PicselliaClassificationPrediction objects.

        Args:
            image_batches (List[List[str]]): A list of image path batches.
            batch_results (List[Results]): A list of inference results corresponding to each batch.
            dataset_context (TDatasetContext): The context of the dataset for processing, including the mapping of labels.

        Returns:
            List[PicselliaClassificationPrediction]: The final classification result, containing predictions for each image,
            including predicted classes and confidence scores.
        """
        all_predictions = []

        for batch_result, batch_paths in zip(batch_results, image_batches):
            all_predictions.extend(
                self._post_process(
                    image_paths=batch_paths,
                    batch_prediction=batch_result,
                    dataset_context=dataset_context,
                )
            )
        return all_predictions

    def _post_process(
        self,
        image_paths: List[str],
        batch_prediction: Results,
        dataset_context: TDatasetContext,
    ) -> List[PicselliaClassificationPrediction]:
        """
        Performs post-processing for a single batch of images and their corresponding predictions.
        Maps the predicted classes and confidence scores to the appropriate Picsellia labels.

        Args:
            image_paths (List[str]): A list of image paths for the batch.
            batch_prediction (Results): The inference results for the batch.
            dataset_context (TDatasetContext): The dataset context used for mapping predicted labels and confidences.

        Returns:
            List[PicselliaClassificationPrediction]: A list of processed results including image paths, predicted classes, and confidence scores.
        """
        processed_predictions = []

        for image_path, prediction in zip(image_paths, batch_prediction):
            asset_id = os.path.basename(image_path).split(".")[0]
            asset = dataset_context.dataset_version.find_all_assets(ids=[asset_id])[0]
            predicted_label = self.get_picsellia_label(
                prediction.names[int(prediction.probs.top1)], dataset_context
            )
            prediction_confidence = self.get_picsellia_confidence(
                float(prediction.probs.top1conf.cpu().numpy())
            )
            processed_prediction = PicselliaClassificationPrediction(
                asset=asset,
                label=predicted_label,
                confidence=prediction_confidence,
            )
            processed_predictions.append(processed_prediction)

        return processed_predictions
