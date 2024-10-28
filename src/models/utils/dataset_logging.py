from typing import Dict
from picsellia import DatasetVersion, Label, Experiment

from picsellia.types.enums import LogType

from picsellia_annotations.coco import COCOFile


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
    experiment.log(
        name=log_name, data=labelmap_to_log, type=LogType.LABELMAP, replace=True
    )


def get_objects_distribution(coco_file: COCOFile) -> Dict:
    """
    Computes the distribution of objects (categories) in a COCO file.

    This function calculates the number of occurrences of each category (object) in a COCO dataset
    by iterating over all annotations and counting the number of times each category appears.

    Args:
        coco_file (COCOFile): The COCO file from which to retrieve the object distribution.

    Returns:
        Dict: A dictionary with two keys, 'x' for category names and 'y' for their corresponding counts.
    """
    objects_distribution = {}
    category_id_to_name = {
        category.id: category.name for category in coco_file.categories
    }
    for annotation in coco_file.annotations:
        category_name = category_id_to_name[annotation.category_id]
        if category_name not in objects_distribution:
            objects_distribution[category_name] = 0
        objects_distribution[category_name] += 1
    return {
        "x": list(objects_distribution.keys()),
        "y": list(objects_distribution.values()),
    }


def log_objects_distribution(
    coco_file: COCOFile, experiment: Experiment, log_name: str
):
    """
    Logs the distribution of objects (categories) in a COCO dataset to an experiment.

    This function computes the object distribution from a COCO file and logs it to the specified
    experiment as a bar chart.

    Args:
        coco_file (COCOFile): The COCO file from which to retrieve the object distribution.
        experiment (Experiment): The experiment where the object distribution will be logged.
        log_name (str): The name under which the object distribution will be logged.
    """
    objects_distribution = get_objects_distribution(coco_file=coco_file)
    experiment.log(
        name=log_name, data=objects_distribution, type=LogType.BAR, replace=True
    )
