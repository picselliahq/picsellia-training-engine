from dataclasses import dataclass
from typing import List

from picsellia import Asset, Label


@dataclass
class PicselliaLabel:
    value: Label


@dataclass
class PicselliaConfidence:
    value: float


@dataclass
class PicselliaRectangle:
    value: List[int]

    def __init__(self, x: int, y: int, w: int, h: int):
        self.value = [x, y, w, h]


@dataclass
class PicselliaText:
    value: str


@dataclass
class PicselliaPolygon:
    value: List[int]


@dataclass
class PicselliaClassificationPrediction:
    asset: Asset
    classes: List[PicselliaLabel]
    confidences: List[PicselliaConfidence]


@dataclass
class PicselliaRectanglePrediction:
    asset: Asset
    boxes: List[PicselliaRectangle]
    classes: List[PicselliaLabel]
    confidences: List[PicselliaConfidence]


@dataclass
class PicselliaOCRPrediction:
    asset: Asset
    boxes: List[PicselliaRectangle]
    classes: List[PicselliaLabel]
    texts: List[PicselliaText]
    confidences: List[PicselliaConfidence]


@dataclass
class PicselliaPolygonPrediction:
    asset: Asset
    polygons: List[PicselliaPolygon]
    classes: List[PicselliaLabel]
    confidences: List[PicselliaConfidence]
