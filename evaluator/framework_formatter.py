from abc import ABC, abstractmethod
from typing import List

from evaluator.utils import (
    cast_type_list_to_float,
    cast_type_list_to_int,
    convert_tensor_to_list,
    rescale_normalized_box,
)
from picsellia.sdk.asset import Asset


class FrameworkFormatter(ABC):
    @classmethod
    @abstractmethod
    def _format_classes(cls, prediction, labelmap) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def _format_confidences(cls, prediction) -> List[float]:
        pass

    @classmethod
    @abstractmethod
    def _format_boxes(cls, asset: Asset, prediction) -> List[int]:
        pass

    @classmethod
    @abstractmethod
    def _format_polygons(cls, prediction):
        pass


class YoloFormatter(FrameworkFormatter):
    @classmethod
    def _format_confidences(cls, prediction) -> List[float]:
        if not prediction.boxes:
            return []
            print("test classification task et si y a pas alors on renvoie []")
        confidences_list = convert_tensor_to_list(tensor=prediction.boxes.conf)
        casted_confidences = cast_type_list_to_float(_list=confidences_list)
        return casted_confidences

    @classmethod
    def _format_classes(cls, prediction, labelmap) -> List[int]:
        if not prediction.boxes:
            return []
            print("test classification task et si y a pas alors on renvoie []")
        classes_list = convert_tensor_to_list(tensor=prediction.boxes.cls)
        casted_classes = cast_type_list_to_int(_list=classes_list)
        picsellia_labels = list(map(lambda label: labelmap[label], casted_classes))
        return picsellia_labels

    @classmethod
    def _format_boxes(cls, asset: Asset, prediction) -> List[int]:
        if not prediction.boxes:
            return []

        normalized_boxes = prediction.boxes.xyxyn
        boxes_list = convert_tensor_to_list(tensor=normalized_boxes)
        rescaled_boxes = list(
            map(
                lambda box: rescale_normalized_box(
                    box=box, width=asset.width, height=asset.height
                ),
                boxes_list,
            )
        )
        casted_boxes = list(map(cast_type_list_to_int, rescaled_boxes))
        return casted_boxes

    @classmethod
    def _format_polygons(cls, prediction):
        if prediction.masks is not None:
            polygons = prediction.masks.xy
            casted_polygons = list(map(lambda polygon: polygon.astype(int), polygons))
            polygons_list = list(map(lambda polygon: polygon.tolist(), casted_polygons))
        else:
            polygons_list = []
        return polygons_list
