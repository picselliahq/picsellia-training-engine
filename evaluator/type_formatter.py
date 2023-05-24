from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from evaluator.framework_formatter import FrameworkFormatter
from picsellia.sdk.asset import Asset
from picsellia.sdk.label import Label
from ultralytics import yolo


class TypeFormatter(ABC):
    def __init__(
        self, framework_formatter: FrameworkFormatter, labelmap: Dict[int, Label]
    ) -> None:
        self._framework_formatter = framework_formatter
        self._labelmap = labelmap

    @abstractmethod
    def _format_predictions(self, asset: Asset, predictions):
        pass

    @abstractmethod
    def _format_evaluation(self, picsellia_prediction):
        pass

    @abstractmethod
    def _get_shape_type(self):
        pass


class ClassificationFormatter(TypeFormatter):
    def _format_predictions(
        self, asset: Asset, prediction
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter._format_confidences(
            confidences=prediction.boxes.conf
        )
        picsellia_predictions["classes"] = self._framework_formatter._format_classes(
            classes=prediction.boxes.cls, labelmap=self._labelmap
        )

        return picsellia_predictions

    def _format_evaluation(self, picsellia_prediction):
        return (picsellia_prediction["classes"], picsellia_prediction["confidences"])

    def _get_shape_type(self):
        return "classifications"


class DetectionFormatter(TypeFormatter):
    def _format_predictions(
        self, asset: Asset, prediction
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions["boxes"] = self._framework_formatter._format_boxes(
            asset=asset, prediction=prediction
        )
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter._format_confidences(prediction=prediction)
        picsellia_predictions["classes"] = self._framework_formatter._format_classes(
            prediction=prediction, labelmap=self._labelmap
        )

        return picsellia_predictions

    def _format_evaluation(self, picsellia_prediction):
        box = picsellia_prediction["boxes"]
        box.append(picsellia_prediction["classes"])
        box.append(picsellia_prediction["confidences"])
        return tuple(box)

    def _get_shape_type(self):
        return "rectangles"


class SegmentationFormatter(TypeFormatter):
    def _format_predictions(
        self, asset: Asset, prediction: yolo.engine.results.Results, get_picsellia_label
    ) -> Tuple[List[float], List[List[int]], List[int]]:
        picsellia_predictions = {}
        picsellia_predictions["polygons"] = self._framework_formatter._format_polygons(
            prediction=prediction
        )
        picsellia_predictions[
            "confidences"
        ] = self._framework_formatter._format_confidences(prediction=prediction)
        picsellia_predictions["classes"] = self._framework_formatter._format_classes(
            prediction=prediction, labelmap=self._labelmap
        )
        return picsellia_predictions

    def _format_evaluation(self, picsellia_prediction):
        return (
            picsellia_prediction["polygons"],
            picsellia_prediction["classes"],
            picsellia_prediction["confidences"],
        )

    def _get_shape_type(self):
        return "polygons"
