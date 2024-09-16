from typing import Optional, Dict, List

import numpy as np
from picsellia import Experiment
from picsellia.types.enums import LogType


class Metric:
    """
    Represents a metric with a standard name and an optional framework-specific name.

    Attributes:
        standard_name (str): The standard name of the metric (e.g., 'accuracy').
        framework_name (Optional[str]): The framework-specific name of the metric (optional).
    """

    def __init__(self, standard_name: str, framework_name: Optional[str] = None):
        """
        Initializes a Metric object.

        Args:
            standard_name (str): The standard name of the metric.
            framework_name (Optional[str]): The framework-specific name of the metric (optional).
        """
        self.standard_name = standard_name
        self.framework_name = framework_name

    def get_name(self) -> str:
        """
        Returns the standard name of the metric.

        Returns:
            str: The standard name of the metric.
        """
        return self.standard_name


class MetricMapping:
    """
    Represents a collection of metric mappings for different phases (train, validation, test) of a model's lifecycle.

    Attributes:
        mappings (Dict[str, List[Metric]]): A dictionary where the key is the phase ('train', 'val', 'test')
                                            and the value is a list of Metric objects.
    """

    def __init__(self):
        """
        Initializes a MetricMapping object with empty lists for train, validation, and test phases.
        """
        self.mappings: Dict[str, List[Metric]] = {
            "train": [],
            "val": [],
            "test": [],
        }

    def add_metric(self, phase: str, metric: Metric) -> None:
        """
        Adds a metric to the specified phase.

        Args:
            phase (str): The phase ('train', 'val', 'test') to which the metric will be added.
            metric (Metric): The metric object to add.

        Raises:
            ValueError: If an unknown phase is provided.
        """
        if phase in self.mappings:
            self.mappings[phase].append(metric)
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def get_mapping(self, phase: Optional[str] = None) -> Dict[str, str]:
        """
        Get the mapping of framework names to standard names for the specified phase.

        Args:
            phase (Optional[str]): The phase ('train', 'val', 'test').

        Returns:
            Dict[str, str]: A dictionary mapping framework names (or standard names) to standard names for the phase.
        """
        if phase is None:
            return {}
        return {
            metric.framework_name or metric.standard_name: metric.standard_name
            for metric in self.mappings.get(phase, [])
        }


class BaseLogger:
    """
    Base class for logging metrics, values, images, and confusion matrices to an experiment.

    Attributes:
        experiment (Experiment): The experiment object for logging.
        metric_mapping (MetricMapping): The metric mapping object to translate metric names.
    """

    def __init__(self, experiment: Experiment, metric_mapping: MetricMapping):
        """
        Initializes the BaseLogger with an experiment and a metric mapping.

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
        Logs a metric value using the experiment's logging system.

        Args:
            name (str): The name of the metric.
            value (float): The value of the metric.
            log_type (LogType): The type of log (e.g., line plot, default is LogType.LINE).
            phase (Optional[str]): The phase in which the metric is logged (e.g., 'train', 'val', 'test').
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, value, log_type)

    def log_value(
        self, name: str, value: float, phase: Optional[str] = None, precision: int = 4
    ):
        """
        Logs a simple scalar value.

        Args:
            name (str): The name of the value.
            value (float): The value to log.
            phase (Optional[str]): The phase in which the value is logged (e.g., 'train', 'val', 'test').
            precision (int): The precision to which the value will be rounded (default is 4).
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, round(value, precision), LogType.VALUE)

    def log_image(self, name: str, image_path: str, phase: Optional[str] = None):
        """
        Logs an image to the experiment.

        Args:
            name (str): The name of the image.
            image_path (str): The path to the image file.
            phase (Optional[str]): The phase in which the image is logged (e.g., 'train', 'val', 'test').
        """
        log_name = self.get_log_name(metric_name=name, phase=phase)
        self.experiment.log(log_name, image_path, LogType.IMAGE)

    def log_confusion_matrix(
        self, labelmap: dict, matrix: np.ndarray, phase: Optional[str] = None
    ):
        """
        Logs a confusion matrix as a heatmap.

        Args:
            labelmap (dict): A mapping of label indices to label names.
            matrix (np.ndarray): The confusion matrix data.
            phase (Optional[str]): The phase in which the confusion matrix is logged (e.g., 'test').
        """
        log_name = self.get_log_name(metric_name="confusion_matrix", phase=phase)
        confusion_data = self._format_confusion_matrix(labelmap, matrix)
        self.experiment.log(log_name, confusion_data, LogType.HEATMAP)

    def _format_confusion_matrix(self, labelmap: dict, matrix: np.ndarray) -> dict:
        """
        Formats the confusion matrix for logging as a heatmap.

        Args:
            labelmap (dict): A mapping of label indices to label names.
            matrix (np.ndarray): The confusion matrix data.

        Returns:
            dict: A dictionary with the categories and matrix values.
        """
        return {"categories": list(labelmap.values()), "values": matrix.tolist()}

    def get_log_name(self, metric_name: str, phase: Optional[str] = None) -> str:
        """
        Generates the appropriate log name by applying the metric mapping and phase.

        Args:
            metric_name (str): The name of the metric.
            phase (Optional[str]): The phase in which the metric is logged (e.g., 'train', 'val', 'test').

        Returns:
            str: The formatted log name.
        """
        mapped_name = self.metric_mapping.get_mapping(phase).get(
            metric_name, metric_name
        )
        return f"{phase}/{mapped_name}" if phase else mapped_name
