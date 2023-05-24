import logging
import math
import os
from abc import ABC, abstractmethod
from typing import List

import tqdm
from evaluator.framework_formatter import FrameworkFormatter
from picsellia.exceptions import (InsufficientResourcesError,
                                  ResourceNotFoundError)
from picsellia.sdk.asset import Asset
from picsellia.sdk.dataset import DatasetVersion
from picsellia.sdk.experiment import Experiment
from picsellia.sdk.label import Label
from evaluator.type_formatter import TypeFormatter


class AbstractEvaluator(ABC):
    type_formatter: TypeFormatter
    framework_formatter: FrameworkFormatter

    def __init__(
        self,
        experiment: Experiment,
        dataset: DatasetVersion,
        asset_list: List[Asset] = None,
    ) -> None:
        self._experiment = experiment
        self._dataset = dataset
        self._parameters = experiment.get_log("parameters").data

        self._asset_list = (
            asset_list if asset_list is not None else self._dataset.list_assets()
        )
        self._batch_size = (
            self._parameters.get("batch_size", 8)
            if len(asset_list) > self._parameters.get("batch_size", 8)
            else len(asset_list)
        )
        self._default_confidence_thresdhold = self._parameters.get(
            "confidence_threshold", 0.1
        )

        self._loaded_model = None
        self._nb_object_limit = 100

        self._labelmap = self._setup_label_map()
        self._formatter = self.type_formatter(
            framework_formatter=self.framework_formatter,
            labelmap=self._labelmap,
        )
        self._setup_evaluation_job()

    def _setup_label_map(self) -> dict[int, Label]:
        experiment_labelmap = self._get_experiment_labelmap()
        dataset_labels = {label.name: label for label in self._dataset.list_labels()}
        self._labels_coherence_check(experiment_labelmap, dataset_labels)

        return {
            int(category_id): dataset_labels[label_name]
            for category_id, label_name in experiment_labelmap.items()
        }

    def _get_experiment_labelmap(self) -> dict:
        try:
            return self._experiment.get_log("labelmap").data
        except Exception:
            raise InsufficientResourcesError(f"Can't find labelmap for this experiment")

    def _labels_coherence_check(self, experiment_labelmap, dataset_labels) -> bool:
        """
        Assert that at least one label from the model labelmap is contained in the dataset version.
        """
        model_labels_name = list(experiment_labelmap.values())
        dataset_labels_name = list(dataset_labels.keys())
        intersecting_labels = set(model_labels_name).intersection(dataset_labels_name)
        logging.info(
            f"Pre-annotation Job will only run on classes: {list(intersecting_labels)}"
        )
        return len(intersecting_labels) > 0

    def _setup_evaluation_job(self):
        logging.info(f"Setting up the evaluation for this experiment")
        self._model_sanity_check()
        self._dataset_inclusion_check()
        self._download_model_weights()
        self._load_saved_model()

    def _model_sanity_check(self) -> None:
        try:
            self._experiment.get_artifact("checkpoint-index-latest")
            logging.info(f"Experiment {self._experiment.name} is sane.")
        except ResourceNotFoundError as e:
            raise ResourceNotFoundError(
                f"Can't run a pre-annotation job with this model, expected a 'checkpoint-index-latest' file"
            ) from e

    def _dataset_inclusion_check(self) -> None:
        """
        Check if the selected dataset is included into the given experiment,

        If the dataset isn't in the experiment, we'll add it under the name "eval".
        """

        attached_datasets = self._experiment.list_attached_dataset_versions()
        for dataset_version in attached_datasets:
            if dataset_version.id == self._dataset.id:
                return

        self._experiment.attach_dataset(name="eval", dataset_version=self._dataset)
        logging.info(
            f"{self._dataset.name}/{self._dataset.version} attached to the experiment."
        )

    def _download_model_weights(self) -> None:
        cwd = os.getcwd()
        model_weights_dir = os.path.join(cwd, "saved_model")
        model_weights = self._experiment.get_artifact(
            self._get_model_artifact_filename()
        )
        model_weights.download(target_path=model_weights_dir)
        self._model_weights_path = os.path.join(
            model_weights_dir, model_weights.filename
        )
        logging.info(f"experiment weights downloaded.")

    @abstractmethod
    def _get_model_artifact_filename(self) -> str:
        pass

    @abstractmethod
    def _load_saved_model(self):
        pass

    def evaluate(self, confidence_threshold: float = None) -> None:
        if confidence_threshold is not None:
            self._confidence_threshold = confidence_threshold
        else:
            self._confidence_threshold = self._default_confidence_thresdhold

        total_batch_number = math.ceil(len(self._asset_list) / self._batch_size)

        for i in tqdm.tqdm(range(total_batch_number)):
            asset_list = self._asset_list[
                i * self._batch_size : (i + 1) * self._batch_size
            ]
            self._evaluate_asset_list(asset_list)
        self._experiment.compute_evaluations_metrics(inference_type=self._dataset.type)

    def _evaluate_asset_list(self, asset_list: List[Asset]) -> None:
        inputs = self._preprocess_images(asset_list)
        predictions = self._loaded_model(inputs)  # Predict
        for i, asset in enumerate(asset_list):
            evaluations = self._format_prediction_to_evaluations(
                asset=asset, prediction=predictions[i]
            )
            self._send_evaluations_to_platform(asset=asset, evaluations=evaluations)

    @abstractmethod
    def _preprocess_images(self):
        pass

    def _format_prediction_to_evaluations(self, asset: Asset, prediction: List) -> List:
        picsellia_predictions = (
            self._formatter._format_predictions(
                asset=asset, prediction=prediction
            )
        )

        evaluations = []
        for i in range(
            min(self._nb_object_limit, len(picsellia_predictions["confidences"]))
        ):
            if picsellia_predictions["confidences"][i] >= self._confidence_threshold:
                picsellia_prediction = {
                    prediction_key: prediction[i]
                    for prediction_key, prediction in picsellia_predictions.items()
                }
                evaluation = self._formatter._format_evaluation(
                    picsellia_prediction=picsellia_prediction
                )
                evaluations.append(evaluation)
        return evaluations

    def _send_evaluations_to_platform(self, asset: Asset, evaluations: List) -> None:
        if len(evaluations) > 0:
            shapes = {self._formatter._get_shape_type(): evaluations}
            self._experiment.add_evaluation(asset=asset, **shapes)
            logging.info(f"Asset: {asset.filename} evaluated.")
        else:
            logging.info(
                f"Asset: {asset.filename} non evaluated, either because the model made no predictions \
                         or because the confidence of the predictions was too low."
            )
