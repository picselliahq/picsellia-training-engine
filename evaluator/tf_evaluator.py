import tensorflow as tf
import requests
from PIL import Image
import logging
import zipfile
import os
from abc import abstractmethod, ABC
from typing import List
import numpy as np

from picsellia.sdk.experiment import Experiment, DatasetVersion
from evaluator.abstract_evaluator import AbstractEvaluator
from evaluator.framework_formatter import TensorflowFormatter
from evaluator.type_formatter import (DetectionFormatter,
                                      SegmentationFormatter, TypeFormatter)
from evaluator.utils import open_asset_as_tensor

from picsellia.exceptions import PicselliaError
from picsellia.sdk.asset import Asset
from functools import partial


class TensorflowEvaluator(AbstractEvaluator):
    framework_formatter = TensorflowFormatter

    @abstractmethod
    def _get_model_artifact_filename(self):
        pass

    def _load_saved_model(self):
        try:
            self._loaded_model = tf.saved_model.load(self._model_weights_path)
            print("Model loaded in memory.")
            try:
                signature = self._loaded_model.signatures[
                    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
                ]
                self.input_width, self.input_height = (
                    signature.inputs[0].shape[1],
                    signature.inputs[0].shape[2],
                )
                self.output_names = list(signature.structured_outputs.keys())
            except Exception as e:
                print(e)
                self.input_width, self.input_height = None, None
                self.output_names = None
        except Exception as e:
            raise PicselliaError(
                f"Impossible to load saved model located at: {self._model_weights_path}"
            )from e

    def _evaluate_asset_list(self, asset_list: List[Asset]) -> None:
        for i, asset in enumerate(asset_list):
            inputs = self._preprocess_image(asset)
            predictions = self._loaded_model(inputs)  # Predict
            evaluations = self._format_prediction_to_evaluations(
                asset=asset, prediction=predictions[i]
            )
            self._send_evaluations_to_platform(asset=asset, evaluations=evaluations)

    def _preprocess_image(self, asset:Asset):
        image = open_asset_as_tensor(asset, self.input_width, self.input_height)
        return image

    def _get_model_weights_path(self):
        weights_zip_path = self._model_weights_path
        with zipfile.ZipFile(weights_zip_path, "r") as zip_ref:
            zip_ref.extractall("saved_model")
        cwd = os.getcwd()
        self._model_weights_path = os.path.join(cwd, "saved_model")


class DetectionTensorflowEvaluator(TensorflowEvaluator):
    type_formatter = DetectionFormatter

    def _get_model_artifact_filename(self):
        return "model-latest"

    def _get_model_task(self):
        return "detect"


class SegmentationTensorflowEvaluator(TensorflowEvaluator):
    type_formatter = SegmentationFormatter

    def _get_model_artifact_filename(self):
        return "model-latest"

    def _get_model_task(self):
        return "segment"
