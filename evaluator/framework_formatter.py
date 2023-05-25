from abc import ABC, abstractmethod
from typing import List

from evaluator.utils import (cast_type_list_to_float, cast_type_list_to_int,
                             convert_tensor_to_list, rescale_normalized_box)
from picsellia.sdk.asset import Asset


class FrameworkFormatter(ABC):
    def __init__(self, labelmap) -> None:
        self._labelmap = labelmap

    @abstractmethod
    def format_classes(self, prediction) -> List[int]:
        pass

    @abstractmethod
    def format_confidences(self, prediction) -> List[float]:
        pass

    @abstractmethod
    def format_boxes(self, asset: Asset, prediction) -> List[int]:
        pass

    @abstractmethod
    def format_polygons(self, prediction):
        pass


class YoloFormatter(FrameworkFormatter):
    def format_confidences(self, prediction) -> List[float]:
        if not prediction.boxes:
            return []
        confidences_list = convert_tensor_to_list(tensor=prediction.boxes.conf)
        casted_confidences = cast_type_list_to_float(_list=confidences_list)
        return casted_confidences

    def format_classes(self, prediction) -> List[int]:
        if not prediction.boxes:
            return []
        classes_list = convert_tensor_to_list(tensor=prediction.boxes.cls)
        casted_classes = cast_type_list_to_int(_list=classes_list)
        picsellia_labels = list(
            map(lambda label: self._labelmap[label], casted_classes)
        )
        return picsellia_labels

    def format_boxes(self, asset: Asset, prediction) -> List[int]:
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

    def format_polygons(self, prediction):
        if prediction.masks is not None:
            polygons = prediction.masks.xy
            casted_polygons = list(map(lambda polygon: polygon.astype(int), polygons))
            polygons_list = list(map(lambda polygon: polygon.tolist(), casted_polygons))
        else:
            polygons_list = []
        return polygons_list
