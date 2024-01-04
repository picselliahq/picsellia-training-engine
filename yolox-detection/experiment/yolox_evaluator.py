import logging
from abc import abstractmethod
from typing import List

import numpy as np
from PIL import UnidentifiedImageError
from picsellia.exceptions import PicselliaError
from picsellia.sdk.asset import Asset

from evaluator.abstract_evaluator import AbstractEvaluator
from evaluator.framework_formatter import YoloFormatter
from evaluator.type_formatter import (
    ClassificationFormatter,
    DetectionFormatter,
    SegmentationFormatter,
    TypeFormatter,
)
from evaluator.utils.general import open_asset_as_array


class YoloxEvaluator(AbstractEvaluator):
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
        images = []
        for asset in assets:
            try:
                image = open_asset_as_array(asset)
            except UnidentifiedImageError:
                logging.warning(
                    f"Can't evaluate {asset.filename}, error opening the image"
                )
                continue
            images.append(image)
        return images

    def _get_model_weights_path(self):
        pass


class ClassificationYoloxEvaluator(YoloxEvaluator):
    type_formatter: TypeFormatter = ClassificationFormatter

    def _get_model_artifact_filename(self):
        return "weights"

    def _get_model_task(self):
        return "classify"


class DetectionYoloxEvaluator(YoloxEvaluator):
    type_formatter: TypeFormatter = DetectionFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "detect"


class SegmentationYoloxEvaluator(YoloxEvaluator):
    type_formatter: TypeFormatter = SegmentationFormatter

    def _get_model_artifact_filename(self):
        return "checkpoint-index-latest"

    def _get_model_task(self):
        return "segment"
