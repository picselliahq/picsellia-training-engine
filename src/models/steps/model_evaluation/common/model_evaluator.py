import logging
from typing import Union, List

from picsellia.types.enums import AddEvaluationType

from src.models.model.common.picsellia_prediction import (
    PicselliaOCRPrediction,
    PicselliaRectanglePrediction,
    PicselliaClassificationPrediction,
    PicselliaPolygonPrediction,
)

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Handles the evaluation process of various prediction types for an experiment in Picsellia.

    The ModelEvaluator class processes different types of predictions (OCR, rectangles, classifications, and polygons)
    and adds them as evaluations to an experiment. The predictions are passed as lists, and each prediction type
    is handled accordingly to log the evaluation results.

    Attributes:
        experiment: The Picsellia experiment to which evaluations will be added.
    """

    def __init__(self, experiment):
        """
        Initializes the ModelEvaluator with the given experiment.

        Args:
            experiment: The Picsellia experiment object where the evaluations will be logged.
        """
        self.experiment = experiment

    def evaluate(
        self,
        picsellia_predictions: Union[
            List[PicselliaClassificationPrediction],
            List[PicselliaRectanglePrediction],
            List[PicselliaPolygonPrediction],
            List[PicselliaOCRPrediction],
        ],
    ) -> None:
        """
        Evaluates a list of predictions and adds them to the experiment.

        Args:
            picsellia_predictions: A list of Picsellia predictions, which can include
                classification, rectangle, polygon, or OCR predictions.
        """
        for prediction in picsellia_predictions:
            self.add_evaluation(prediction)

    def add_evaluation(
        self,
        evaluation: Union[
            PicselliaClassificationPrediction,
            PicselliaRectanglePrediction,
            PicselliaPolygonPrediction,
            PicselliaOCRPrediction,
        ],
    ) -> None:
        """
        Adds a single evaluation to the experiment based on the prediction type.

        Args:
            evaluation: A single prediction instance, which can be a classification, rectangle, polygon, or OCR prediction.

        Raises:
            TypeError: If the prediction type is not supported.
        """
        asset = evaluation.asset

        if isinstance(evaluation, PicselliaOCRPrediction):
            rectangles = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                )
                for rectangle, label, conf in zip(
                    evaluation.boxes, evaluation.labels, evaluation.confidences
                )
            ]
            rectangles_with_text = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                    text.value,
                )
                for rectangle, label, conf, text in zip(
                    evaluation.boxes,
                    evaluation.labels,
                    evaluation.confidences,
                    evaluation.texts,
                )
            ]
            logger.info(
                f"Adding evaluation for asset {asset.filename} with rectangles {rectangles_with_text}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, rectangles=rectangles
            )

        elif isinstance(evaluation, PicselliaRectanglePrediction):
            rectangles = [
                (
                    rectangle.value[0],
                    rectangle.value[1],
                    rectangle.value[2],
                    rectangle.value[3],
                    label.value,
                    conf.value,
                )
                for rectangle, label, conf in zip(
                    evaluation.boxes, evaluation.labels, evaluation.confidences
                )
            ]
            logger.info(
                f"Adding evaluation for asset {asset.filename} with rectangles {rectangles}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, rectangles=rectangles
            )

        elif isinstance(evaluation, PicselliaClassificationPrediction):
            classifications = [(evaluation.label.value, evaluation.confidence.value)]
            self.experiment.add_evaluation(
                asset,
                add_type=AddEvaluationType.REPLACE,
                classifications=classifications,
            )

        elif isinstance(evaluation, PicselliaPolygonPrediction):
            polygons = [
                (polygon.value, label.value, conf.value)
                for polygon, label, conf in zip(
                    evaluation.polygons, evaluation.labels, evaluation.confidences
                )
            ]
            logger.info(
                f"Adding evaluation for asset {asset.filename} with polygons {polygons}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, polygons=polygons
            )

        else:
            raise TypeError("Unsupported prediction type")
