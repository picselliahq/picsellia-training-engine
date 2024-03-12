import os

from picsellia import DatasetVersion

from poc.step import step


def dataset_extractor(context: dict, dataset: DatasetVersion) -> dict:
    dataset_extraction_path = os.path.join(context["experiment"].name, dataset.name)
    dataset.download(
        target_path=os.path.join(dataset_extraction_path, "images", dataset.version),
        use_id=True,
    )
    coco_file = dataset.build_coco_file_locally(use_id=True)
    labelmap = {label.name: label for label in dataset.list_labels()}
    dataset_context = {
        "dataset_extraction_path": dataset_extraction_path,
        "images_dir": os.path.join(dataset_extraction_path, "images", dataset.version),
        "coco_file": coco_file,
        "labelmap": labelmap,
        "images_list": [
            os.path.join(
                os.path.join(dataset_extraction_path, "images", dataset.version), img
            )
            for img in os.listdir(
                os.path.join(dataset_extraction_path, "images", dataset.version)
            )
        ],
    }
    return dataset_context


@step
def data_extractor(context: dict):
    train_dataset: DatasetVersion = context["experiment"].get_dataset("train")
    test_dataset: DatasetVersion = context["experiment"].get_dataset("test")
    train_dataset_context = dataset_extractor(context, train_dataset)
    test_dataset_context = dataset_extractor(context, test_dataset)
    return {"train": train_dataset_context, "test": test_dataset_context}
