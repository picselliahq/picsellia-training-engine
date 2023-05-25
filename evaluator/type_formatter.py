from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Type

from evaluator.framework_formatter import FrameworkFormatter
from picsellia.sdk.asset import Asset
from picsellia.sdk.label import Label


class TypeFormatter(ABC):
    def __init__(self, framework_formatter: FrameworkFormatter) -> None:
        self._framework_formatter = framework_formatter

    @abstractmethod
    def format_predictions(self, asset: Asset, predictions):
        pass

    @abstractmethod
    def format_evaluation(self, picsellia_prediction):
        pass

    @abstractmethod
    def get_shape_type(self):
        pass


class ClassificationFormatter(TypeFormatter):
    def format_predictions(
        self, asset: Asset, prediction
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter.format_confidences(
            confidences=prediction.boxes.conf
        )
        picsellia_predictions["classes"] = self._framework_formatter.format_classes(
            classes=prediction.boxes.cls
        )

        return picsellia_predictions

    def format_evaluation(self, picsellia_prediction):
        return (picsellia_prediction["classes"], picsellia_prediction["confidences"])

    def get_shape_type(self):
        return "classifications"


class DetectionFormatter(TypeFormatter):
    def format_predictions(
        self, asset: Asset, prediction
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions["boxes"] = self._framework_formatter.format_boxes(
            asset=asset, prediction=prediction
        )
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter.format_confidences(prediction=prediction)
        picsellia_predictions["classes"] = self._framework_formatter.format_classes(
            prediction=prediction
        )

        return picsellia_predictions

    def format_evaluation(self, picsellia_prediction):
        box = picsellia_prediction["boxes"]
        box.append(picsellia_prediction["classes"])
        box.append(picsellia_prediction["confidences"])
        return tuple(box)

    def get_shape_type(self):
        return "rectangles"


class SegmentationFormatter(TypeFormatter):
    def format_predictions(
        self, asset: Asset, prediction
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions["polygons"] = self._framework_formatter.format_polygons(
            prediction=prediction
        )
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter.format_confidences(prediction=prediction)
        picsellia_predictions["classes"] = self._framework_formatter.format_classes(
            prediction=prediction
        )
        return picsellia_predictions

    def format_evaluation(self, picsellia_prediction):
        return (
            picsellia_prediction["polygons"],
            picsellia_prediction["classes"],
            picsellia_prediction["confidences"],
        )

    def get_shape_type(self):
        return "polygons"
