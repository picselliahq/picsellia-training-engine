from typing import Dict
from picsellia import DatasetVersion, Label, Experiment

from picsellia.types.enums import LogType

from picsellia_annotations.coco import COCOFile


def get_labelmap(dataset_version: DatasetVersion) -> Dict[str, Label]:
    """
    Retrieves the label map from a dataset version.

    Parameters:
        dataset_version (DatasetVersion): The dataset version from which to retrieve the label map.

    Returns:
        Dict[str, Label]: A dictionary mapping label names to label objects.
    """
    return {label.name: label for label in dataset_version.list_labels()}


def log_labelmap(labelmap: Dict[str, Label], experiment: Experiment, log_name: str):
    """
    Logs the label map to the experiment.

    Parameters:
        labelmap (Dict[str, Label]): A dictionary mapping label names to label objects.
        experiment (Experiment): The experiment object where the label map is logged.
    """
    labelmap_to_log = {str(i): label for i, label in enumerate(labelmap.keys())}
    experiment.log(
        name=log_name, data=labelmap_to_log, type=LogType.TABLE, replace=True
    )


def get_objects_distribution(coco_file: COCOFile) -> Dict:
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
    objects_distribution = get_objects_distribution(coco_file=coco_file)
    experiment.log(
        name=log_name, data=objects_distribution, type=LogType.BAR, replace=True
    )
