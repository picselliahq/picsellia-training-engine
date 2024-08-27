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
        self.value = [int(x), int(y), int(w), int(h)]


@dataclass
class PicselliaText:
    value: str


@dataclass
class PicselliaPolygon:
    value: List[int]

    def __init__(self, points: List[int]):
        self.value = [int(point) for point in points]


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


class PredictionRectangleResult:
    def __init__(
        self,
        image_paths: List[str],
        boxes: List[List[PicselliaRectangle]],
        labels: List[List[PicselliaLabel]],
        confidences: List[List[PicselliaConfidence]],
    ):
        self.image_paths = image_paths
        self.boxes = boxes
        self.labels = labels
        self.confidences = confidences

    def __getitem__(self, idx):
        return {
            "image_path": self.image_paths[idx],
            "boxes": self.boxes[idx],
            "labels": self.labels[idx],
            "confidences": self.confidences[idx],
        }

    def __len__(self):
        return len(self.image_paths)


class PredictionPolygonResult:
    def __init__(
        self,
        image_paths: List[str],
        polygons: List[List[PicselliaPolygon]],
        labels: List[List[PicselliaLabel]],
        confidences: List[List[PicselliaConfidence]],
    ):
        self.image_paths = image_paths
        self.polygons = polygons
        self.labels = labels
        self.confidences = confidences

    def __getitem__(self, idx):
        return {
            "image_path": self.image_paths[idx],
            "polygons": self.polygons[idx],
            "labels": self.labels[idx],
            "confidences": self.confidences[idx],
        }

    def __len__(self):
        return len(self.image_paths)


class PredictionOCRResult:
    def __init__(
        self,
        image_paths: List[str],
        boxes: List[List[PicselliaRectangle]],
        labels: List[List[PicselliaLabel]],
        texts: List[List[PicselliaText]],
        confidences: List[List[PicselliaConfidence]],
    ):
        self.image_paths = image_paths
        self.boxes = boxes
        self.labels = labels
        self.texts = texts
        self.confidences = confidences

    def __getitem__(self, idx):
        return {
            "image_path": self.image_paths[idx],
            "boxes": self.boxes[idx],
            "labels": self.labels[idx],
            "texts": self.texts[idx],
            "confidences": self.confidences[idx],
        }

    def __len__(self):
        return len(self.image_paths)


class PredictionClassificationResult:
    def __init__(
        self,
        image_paths: List[str],
        classes: List[List[PicselliaLabel]],
        confidences: List[List[PicselliaConfidence]],
    ):
        self.image_paths = image_paths
        self.classes = classes
        self.confidences = confidences

    def __getitem__(self, idx):
        return {
            "image_path": self.image_paths[idx],
            "classes": self.classes[idx],
            "confidences": self.confidences[idx],
        }

    def __len__(self):
        return len(self.image_paths)
