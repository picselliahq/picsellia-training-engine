import os
import shutil

from src import step, Pipeline
from picsellia.types.enums import AnnotationFileType

from src.models.dataset.common.yolov7_dataset_collection import Yolov7DatasetCollection

BATCH_SIZE = 10000


@step
def yolov7_dataset_collection_preparator(
    dataset_collection: Yolov7DatasetCollection,
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
        os.makedirs(
            os.path.join(images_destination_path, dataset_context.dataset_name),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(annotations_destination_path, dataset_context.dataset_name),
            exist_ok=True,
        )

        for i in range(0, len(dataset_context.assets), BATCH_SIZE):
            batch_assets = dataset_context.assets[i : i + BATCH_SIZE]

            yolo_annotation_path = (
                dataset_context.dataset_version.export_annotation_file(
                    annotation_file_type=AnnotationFileType.YOLO,
                    target_path=os.path.join(
                        yolov7_dataset_collection.dataset_path,
                        dataset_context.dataset_name,
                        "annotations",
                    ),
                    use_id=True,
                    assets=batch_assets,
                )
            )

            os.system(
                f"unzip -o {yolo_annotation_path} -d {os.path.join(annotations_destination_path, dataset_context.dataset_name)} > /dev/null 2>&1"
            )
            # os.remove(yolo_annotation_path)

        image_files = os.listdir(dataset_context.images_dir)
        destination_dir = os.path.join(
            images_destination_path, dataset_context.dataset_name
        )

        for i in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[i : i + BATCH_SIZE]
            for file_name in batch_files:
                src_file = os.path.join(dataset_context.images_dir, file_name)
                shutil.copy(src_file, destination_dir)

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
