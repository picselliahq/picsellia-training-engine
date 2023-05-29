import logging
from abc import abstractmethod
from typing import List

import numpy as np
from evaluator.abstract_evaluator import AbstractEvaluator
from evaluator.framework_formatter import YoloFormatter
from evaluator.type_formatter import (ClassificationFormatter,
                                      DetectionFormatter,
                                      SegmentationFormatter, TypeFormatter)
from evaluator.utils import open_asset_as_array
from picsellia.exceptions import PicselliaError
from picsellia.sdk.asset import Asset
from ultralytics import YOLO


class YOLOEvaluator(AbstractEvaluator):
    framework_formatter = YoloFormatter

    @abstractmethod
    def _get_model_artifact_filename(self):
        pass

    def _load_saved_model(self):
        try:
            self._loaded_model = YOLO(
                self._model_weights_path, task=self._get_model_task()
            )
            logging.info("Model loaded in memory.")
        except Exception as e:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self._model_weights_path}"
            ) from e

    @abstractmethod
    def _get_model_task(self):
        pass

    def _preprocess_images(self, assets: List[Asset]) -> List[np.array]:
        images = list(map(open_asset_as_array, assets))
        return images


class ClassificationYOLOEvaluator(YOLOEvaluator):
    type_formatter: TypeFormatter = ClassificationFormatter

    def _get_model_artifact_filename(self):
        return "weights"

    def _get_model_task(self):
        return "classify"


class DetectionYOLOEvaluator(YOLOEvaluator):
    type_formatter: TypeFormatter = DetectionFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "detect"


class SegmentationYOLOEvaluator(YOLOEvaluator):
    type_formatter: TypeFormatter = SegmentationFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "segment"
