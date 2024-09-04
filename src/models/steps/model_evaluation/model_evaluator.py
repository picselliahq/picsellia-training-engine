from typing import Union, List

from picsellia.types.enums import AddEvaluationType

from src.models.model.common.picsellia_prediction import (
    PicselliaOCRPrediction,
    PicselliaRectanglePrediction,
    PicselliaClassificationPrediction,
    PicselliaPolygonPrediction,
)


class ModelEvaluator:
    def __init__(self, experiment):
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
