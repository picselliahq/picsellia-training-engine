from typing import Optional

import numpy as np
from picsellia import Experiment
from picsellia.types.enums import LogType


# class MetricStrategy:
#     def __init__(self, phase: str, metric_name: str, value: float):
#         self.phase = phase
#         self.metric_name = metric_name
#         self.value = value
#
#     def log(self, logger: "BaseLogger"):
#         """
#         Détermine automatiquement comment loguer la métrique en fonction de la phase et du type de métrique.
#         """
#         if self.phase == "train":
#             if "loss" in self.metric_name:
#                 logger.log_metric(self.phase, self.metric_name, self.value)
#             elif "accuracy" in self.metric_name:
#                 logger.log_metric(self.phase, self.metric_name, self.value)
#                 # On peut aussi loguer l'accuracy finale en tant que value
#                 logger.log_value(f"{self.phase}/final_accuracy", self.value)
#             elif "epoch_time" in self.metric_name:
#                 logger.log_metric(self.phase, self.metric_name, self.value)
#                 # Supposons qu'on calcule l'average epoch time
#                 logger.log_value(
#                     f"{self.phase}/average_epoch_time", np.mean(self.value)
#                 )
#         elif self.phase == "val":
#             if "accuracy" in self.metric_name or "loss" in self.metric_name:
#                 logger.log_metric(self.phase, self.metric_name, self.value)
#         elif self.phase == "test":
#             # Sur la phase test, on logue les valeurs finales
#             logger.log_value(f"{self.phase}/{self.metric_name}", self.value)
#         else:
#             raise ValueError(f"Phase {self.phase} non reconnue pour le logging.")


class BaseLogger:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.metric_mappings = self._get_default_metric_mappings()

    def _get_default_metric_mappings(self) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses")

    def log_metric(
        self,
        name: str,
        value: float,
        log_type: LogType = LogType.LINE,
        phase: Optional[str] = None,
    ):
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, value, log_type)

    def log_value(self, name: str, value: float, phase: Optional[str] = None):
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, value, LogType.VALUE)

    def log_image(self, name: str, image_path: str, phase: Optional[str] = None):
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, image_path, LogType.IMAGE)

    def log_confusion_matrix(
        self, labelmap: dict, matrix: np.ndarray, phase: Optional[str] = None
    ):
        log_name = self.get_log_name(metric_name="confusion_matrix", phase=phase)
        confusion_data = self._format_confusion_matrix(labelmap, matrix)
        self.experiment.log(log_name, confusion_data, LogType.HEATMAP)

    def _format_confusion_matrix(self, labelmap: dict, matrix: np.ndarray) -> dict:
        return {"categories": list(labelmap.values()), "values": matrix.tolist()}

    def get_log_name(self, metric_name: str, phase: Optional[str] = None) -> str:
        mapped_name = self.metric_mappings.get(phase, {}).get(metric_name, metric_name)
        return f"{phase}/{mapped_name}" if phase else mapped_name

    # def log(self, phase: str, metric_name: str, value: float):
    #     strategy = MetricStrategy(phase, metric_name, value)
    #     strategy.log(self)
