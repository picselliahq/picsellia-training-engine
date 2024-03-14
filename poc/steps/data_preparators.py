import os

from poc.models.dataset.dataset_organizer import DatasetOrganizer
from poc.step import step


@step
def data_preparator(dataset_context: dict):
    train_converter = DatasetOrganizer(
        dataset_context["train"]["coco_file"],
        dataset_context["train"]["images_dir"],
        os.path.join(dataset_context["train"]["dataset_extraction_path"], "train"),
    )
    train_converter.organize()

    test_converter = DatasetOrganizer(
        dataset_context["test"]["coco_file"],
        dataset_context["test"]["images_dir"],
        os.path.join(dataset_context["test"]["dataset_extraction_path"], "test"),
    )
    test_converter.organize()

    if (
        dataset_context["train"]["dataset_extraction_path"]
        == dataset_context["test"]["dataset_extraction_path"]
    ):
        return os.path.abspath(dataset_context["train"]["dataset_extraction_path"])
    else:
        raise ValueError("Train and test global dataset paths are different")
