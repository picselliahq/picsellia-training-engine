import os
from src import step, Pipeline
from src.models.dataset.common.dataset_collection import DatasetCollection
from picsellia.types.enums import AnnotationFileType

from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection


@step
def yolov7_dataset_collection_preparator(
    dataset_collection: DatasetCollection,
) -> Yolov7DatasetCollection:
    context = Pipeline.get_active_context()

    yolov7_dataset_collection = Yolov7DatasetCollection(
        datasets=list(dataset_collection.datasets.values())
    )

    yolov7_dataset_collection.dataset_path = os.path.join(
        os.getcwd(), context.experiment.name, "yolov7_dataset"
    )
    images_destination_path = os.path.join(
        yolov7_dataset_collection.dataset_path, "images"
    )
    annotations_destination_path = os.path.join(
        yolov7_dataset_collection.dataset_path, "labels"
    )

    for dataset_context in yolov7_dataset_collection:
        # Create directories if they do not exist
        os.makedirs(
            os.path.join(images_destination_path, dataset_context.dataset_name),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(annotations_destination_path, dataset_context.dataset_name),
            exist_ok=True,
        )

        yolo_annotation_path = dataset_context.dataset_version.export_annotation_file(
            annotation_file_type=AnnotationFileType.YOLO,
            target_path=dataset_context.annotations_dir,
            use_id=True,
        )
        os.system(
            f"unzip -o {yolo_annotation_path} -d {os.path.join(annotations_destination_path, dataset_context.dataset_name)} > /dev/null 2>&1"
        )
        # Remove the zip file
        # os.system(f"rm {yolo_annotation_path}")

        # Move images to the destination path
        os.system(
            f"cp {dataset_context.images_dir}/* {os.path.join(images_destination_path, dataset_context.dataset_name)}"
        )

        # Update paths in dataset context
        dataset_context.images_dir = os.path.join(
            images_destination_path, dataset_context.dataset_name
        )
        dataset_context.annotations_dir = os.path.join(
            annotations_destination_path, dataset_context.dataset_name
        )

    yolov7_dataset_collection.write_config(
        os.path.join(yolov7_dataset_collection.dataset_path, "dataset_config.yaml")
    )

    return yolov7_dataset_collection
