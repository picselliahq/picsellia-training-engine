from typing import Dict
from picsellia import DatasetVersion, Label, Experiment

from picsellia.types.enums import LogType
from picsellia.exceptions import ResourceNotFoundError


def get_labelmap(dataset_version: DatasetVersion) -> Dict[str, Label]:
    """
    Retrieves the label map from a dataset version.

    This function generates a dictionary that maps label names to their corresponding
    label objects from a given dataset version.

    Args:
        dataset_version (DatasetVersion): The dataset version from which to retrieve the label map.

    Returns:
        Dict[str, Label]: A dictionary mapping label names to their corresponding Label objects.
    """
    return {label.name: label for label in dataset_version.list_labels()}


def log_labelmap(labelmap: Dict[str, Label], experiment: Experiment, log_name: str):
    """
    Logs the label map to an experiment.

    This function logs the label map to a specified experiment in a tabular format.

    Args:
        labelmap (Dict[str, Label]): A dictionary mapping label names to Label objects.
        experiment (Experiment): The experiment where the label map will be logged.
        log_name (str): The name under which the label map will be logged.
    """
    labelmap_to_log = {str(i): label for i, label in enumerate(labelmap.keys())}
    try:
        picsellia_labelmap = experiment.get_log(name=log_name)
        picsellia_labelmap.update(data=labelmap_to_log)
    except ResourceNotFoundError:
        experiment.log(
            name=log_name, data=labelmap_to_log, type=LogType.LABELMAP, replace=True
        )
