import os

from src import step

from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection

BATCH_SIZE = 10000


@step
def yolov7_dataset_collection_preparator(
    dataset_collection: Yolov7DatasetCollection,
) -> Yolov7DatasetCollection:
    if not dataset_collection.dataset_path:
        raise ValueError("Dataset path is not set in the dataset collection.")
    dataset_collection.write_config(
        config_path=os.path.join(dataset_collection.dataset_path, "dataset_config.yaml")
    )
    return dataset_collection
