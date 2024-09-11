from typing import Optional, Dict, List

import numpy as np
from picsellia import Experiment
from picsellia.types.enums import LogType


class Metric:
    def __init__(self, standard_name: str, framework_name: Optional[str] = None):
        self.standard_name = standard_name
        self.framework_name = framework_name

    def get_name(self) -> str:
        return self.standard_name


class MetricMapping:
    """
    Represents a collection of metrics mappings for different phases of a model's lifecycle.
    """

    def __init__(self):
        self.mappings: Dict[str, List[Metric]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    def add_metric(self, phase: str, metric: Metric) -> None:
        """
        Adds a metric to the specified phase.

        Args:
            phase (str): The phase ('train', 'val', 'test').
            metric (Metric): The metric object to add.
        """
        if phase in self.mappings:
            self.mappings[phase].append(metric)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_mapping(self, phase: Optional[str] = None) -> Dict[str, str]:
        """
        Get the mapping for the specified phase.

        Args:
            phase: The phase ('train', 'val', 'test').

        Returns:
            A dictionary mapping framework names to standard names.
        """
        if phase is None:
            return {}
        return {
            metric.framework_name or metric.standard_name: metric.standard_name
            for metric in self.mappings.get(phase, [])
        }


class ClassificationMetricMapping(MetricMapping):
    def __init__(self):
        super().__init__()
        self.add_metric(phase="train", metric=Metric("accuracy"))
        self.add_metric(phase="train", metric=Metric("loss"))
        self.add_metric(phase="train", metric=Metric("learning_rate"))
        self.add_metric(phase="train", metric=Metric("epoch_time"))

        self.add_metric(phase="val", metric=Metric("accuracy"))
        self.add_metric(phase="val", metric=Metric("loss"))

        self.add_metric(phase="test", metric=Metric("accuracy"))
        self.add_metric(phase="test", metric=Metric("precision"))
        self.add_metric(phase="test", metric=Metric("recall"))
        self.add_metric(phase="test", metric=Metric("f1_score"))
        self.add_metric(phase="test", metric=Metric("loss"))
        self.add_metric(phase="test", metric=Metric("confusion_matrix"))


class BaseLogger:
    def __init__(self, experiment: Experiment, metric_mapping: MetricMapping):
        """
        Initialize the logger with an experiment and a metric mapping.

        Args:
            experiment (Experiment): The experiment object for logging.
            metric_mapping (MetricMapping): The metric mapping object to translate metric names.
        """
        self.experiment = experiment
        self.metric_mapping = metric_mapping

    def log_metric(
        self,
        name: str,
        value: float,
        log_type: LogType = LogType.LINE,
        phase: Optional[str] = None,
    ):
        """
        Log a metric value using the experiment's logging system.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
            log_type (LogType): The type of log (e.g., line plot).
            phase (Optional[str]): The phase (e.g., 'train', 'val', 'test').
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, value, log_type)

    def log_value(
        self, name: str, value: float, phase: Optional[str] = None, precision: int = 4
    ):
        """
        Log a simple value.

        Args:
            name (str): The name of the value.
            value (float): The value to log.
            phase (Optional[str]): The phase (e.g., 'train', 'val', 'test').
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, round(value, precision), LogType.VALUE)

    def log_image(self, name: str, image_path: str, phase: Optional[str] = None):
        """
        Log an image.

        Args:
            name (str): The name of the image.
            image_path (str): The path to the image file.
            phase (Optional[str]): The phase (e.g., 'train', 'val', 'test').
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, image_path, LogType.IMAGE)

    def log_confusion_matrix(
        self, labelmap: dict, matrix: np.ndarray, phase: Optional[str] = None
    ):
        """
        Log a confusion matrix as a heatmap.

        Args:
            labelmap (dict): A mapping of label indices to label names.
            matrix (np.ndarray): The confusion matrix.
            phase (Optional[str]): The phase (e.g., 'test').
        """
        log_name = self.get_log_name(metric_name="confusion_matrix", phase=phase)
        confusion_data = self._format_confusion_matrix(labelmap, matrix)
        self.experiment.log(log_name, confusion_data, LogType.HEATMAP)

    def _format_confusion_matrix(self, labelmap: dict, matrix: np.ndarray) -> dict:
        """
        Format the confusion matrix for logging.

        Args:
            labelmap (dict): A mapping of label indices to label names.
            matrix (np.ndarray): The confusion matrix.

        Returns:
            dict: A formatted confusion matrix.
        """
        return {"categories": list(labelmap.values()), "values": matrix.tolist()}

    def get_log_name(self, metric_name: str, phase: Optional[str] = None) -> str:
        """
        Get the log name after applying metric mapping.

        Args:
            metric_name (str): The standard name of the metric.
            phase (Optional[str]): The phase (e.g., 'train', 'val', 'test').

        Returns:
            str: The mapped name of the metric for logging.
        """
        mapped_name = self.metric_mapping.get_mapping(phase).get(
            metric_name, metric_name
        )
        return f"{phase}/{mapped_name}" if phase else mapped_name
