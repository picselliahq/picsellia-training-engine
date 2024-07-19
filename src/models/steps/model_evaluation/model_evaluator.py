from abc import ABC
from typing import TypeVar, Generic

from picsellia import Experiment
from picsellia.types.enums import AddEvaluationType

from src.models.dataset.common.dataset_collection import TDatasetContext
from src.models.model.picsellia_prediction import (
    PicselliaOCRPrediction,
    PicselliaRectanglePrediction,
    PicselliaClassificationPrediction,
    PicselliaPolygonPrediction,
)
from src.models.steps.model_inferencing.base_model_inference import BaseModelInference

TModelInference = TypeVar("TModelInference", bound=BaseModelInference)


class ModelEvaluator(ABC, Generic[TModelInference]):
    def __init__(
        self,
        model_inference: TModelInference,
        dataset_context: TDatasetContext,
        experiment: Experiment,
    ):
        self.model_inference = model_inference
        self.dataset_context = dataset_context
        self.experiment = experiment

    def evaluate(self):
        picsellia_evaluations = self.model_inference.predict_on_dataset_context(
            self.dataset_context
        )
        for evaluation in picsellia_evaluations:
            self.add_evaluation(evaluation)

    def add_evaluation(self, evaluation):
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
                    evaluation.boxes, evaluation.classes, evaluation.confidences
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
                    evaluation.classes,
                    evaluation.confidences,
                    evaluation.texts,
                )
            ]
            print(
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
                    evaluation.boxes, evaluation.classes, evaluation.confidences
                )
            ]
            print(
                f"Adding evaluation for asset {asset.filename} with rectangles {rectangles}"
            )
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, rectangles=rectangles
            )

        elif isinstance(evaluation, PicselliaClassificationPrediction):
            classifications = [
                (label.value, conf.value)
                for label, conf in zip(evaluation.classes, evaluation.confidences)
            ]
            self.experiment.add_evaluation(
                asset,
                add_type=AddEvaluationType.REPLACE,
                classifications=classifications,
            )

        elif isinstance(evaluation, PicselliaPolygonPrediction):
            polygons = [
                (polygon.value, label.value, conf.value)
                for polygon, label, conf in zip(
                    evaluation.polygons, evaluation.classes, evaluation.confidences
                )
            ]
            self.experiment.add_evaluation(
                asset, add_type=AddEvaluationType.REPLACE, polygons=polygons
            )

        else:
            raise TypeError("Unsupported prediction type")
