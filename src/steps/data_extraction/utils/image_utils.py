from typing import Dict

from picsellia import DatasetVersion, Label


def get_labelmap(dataset_version: DatasetVersion) -> Dict[str, Label]:
    """
    Retrieves the label map from a dataset version.

    Parameters:
        dataset_version (DatasetVersion): The dataset version from which to retrieve the label map.

    Returns:
        Dict: A dictionary mapping label names to label objects.
    """
    return {label.name: label for label in dataset_version.list_labels()}
