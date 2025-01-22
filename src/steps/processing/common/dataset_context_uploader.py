import json
import os

from picsellia.types.enums import InferenceType

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.coco_dataset_context import CocoDatasetContext
from src.models.steps.processing.common.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)
from src.models.steps.processing.common.data_uploader import DataUploader
from src.models.steps.processing.common.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)
from src.models.steps.processing.common.segmentation_dataset_context_uploader import (
    SegmentationDatasetContextUploader,
)


@step
def upload_dataset_context(
    dataset_context: CocoDatasetContext,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    if dataset_context.coco_data and not dataset_context.coco_file_path:
        if not dataset_context.annotations_dir:
            dataset_context.annotations_dir = "temp_annotations"
            os.makedirs(dataset_context.annotations_dir, exist_ok=True)
        dataset_context.coco_file_path = os.path.join(
            dataset_context.annotations_dir, "annotations.json"
        )
        with open(dataset_context.coco_file_path, "w") as f:
            json.dump(dataset_context.coco_data, f)
    if dataset_context.dataset_version.type == InferenceType.NOT_CONFIGURED:
        if not dataset_context.coco_data["annotations"]:
            simple_uploader = DataUploader(
                client=context.client,
                dataset_version=dataset_context.dataset_version,
                datalake=context.processing_parameters.datalake,
            )
            if dataset_context.images_dir:
                simple_uploader._add_images_to_dataset_version_in_batches(
                    images_to_upload=[
                        os.path.join(dataset_context.images_dir, image_filename)
                        for image_filename in os.listdir(dataset_context.images_dir)
                    ],
                    data_tags=[
                        context.processing_parameters.data_tag,
                        dataset_context.dataset_version.version,
                    ],
                )
        elif dataset_context.coco_data["annotations"][0]["segmentation"]:
            dataset_context.dataset_version.set_type(InferenceType.SEGMENTATION)
        elif dataset_context.coco_data["annotations"][0]["bbox"]:
            dataset_context.dataset_version.set_type(InferenceType.OBJECT_DETECTION)
        elif dataset_context.coco_data["annotations"][0]["category_id"]:
            dataset_context.dataset_version.set_type(InferenceType.CLASSIFICATION)
        else:
            raise ValueError(
                f"Unsupported dataset type: {dataset_context.dataset_version.type}"
            )
    if dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetContextUploader(
            client=context.client,
            dataset_context=dataset_context,
            datalake=context.processing_parameters.datalake,
            data_tags=[
                context.processing_parameters.data_tag,
                dataset_context.dataset_version.version,
            ],
        )
        classification_uploader.upload_dataset_context()
    elif dataset_context.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetContextUploader(
            client=context.client,
            dataset_context=dataset_context,
            datalake=context.processing_parameters.datalake,
            data_tags=[
                context.processing_parameters.data_tag,
                dataset_context.dataset_version.version,
            ],
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
        object_detection_uploader.upload_dataset_context()
    elif dataset_context.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetContextUploader(
            client=context.client,
            dataset_context=dataset_context,
            datalake=context.processing_parameters.datalake,
            data_tags=[
                context.processing_parameters.data_tag,
                dataset_context.dataset_version.version,
            ],
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
        segmentation_uploader.upload_dataset_context()
