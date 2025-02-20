import json
import os
from typing import Optional, Dict, Any, List

from picsellia.types.enums import InferenceType

from src.picsellia_cv_engine import Pipeline
from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)

from picsellia import Client

from src.picsellia_cv_engine.models.steps.data_upload.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)
from src.picsellia_cv_engine.models.steps.data_upload.data_uploader import DataUploader
from src.picsellia_cv_engine.models.steps.data_upload.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)
from src.picsellia_cv_engine.models.steps.data_upload.segmentation_dataset_context_uploader import (
    SegmentationDatasetContextUploader,
)


@step
def upload_dataset_context(
    dataset_context: CocoDatasetContext,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
    client: Optional[Client] = None,
    datalake: Optional[str] = None,
    data_tag: Optional[str] = None,
) -> None:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    if context:
        client = context.client
        datalake = context.processing_parameters.datalake
        data_tag = context.processing_parameters.data_tag
    elif not all([client, datalake, data_tag]):
        raise ValueError(
            "One of the following parameter sets must be provided: either context, or (client, datalake, data_tag)."
        )

    if not datalake or not data_tag:
        raise ValueError("datalake and data_tag must not be None")

    if dataset_context.coco_data and not dataset_context.coco_file_path:
        if not dataset_context.annotations_dir:
            dataset_context.annotations_dir = "temp_annotations"
            os.makedirs(dataset_context.annotations_dir, exist_ok=True)
        dataset_context.coco_file_path = os.path.join(
            dataset_context.annotations_dir, "annotations.json"
        )
        with open(dataset_context.coco_file_path, "w") as f:
            json.dump(dataset_context.coco_data, f)

    if dataset_context.coco_file_path and not dataset_context.coco_data:
        dataset_context.coco_data = dataset_context.load_coco_file_data()

    coco_data: Dict[str, Any] = (
        dataset_context.coco_data if dataset_context.coco_data else {}
    )

    # Configure dataset type if not already set
    if dataset_context.dataset_version.type == InferenceType.NOT_CONFIGURED:
        annotations = coco_data.get("annotations", [])

        if not annotations:
            if dataset_context.images_dir:
                simple_uploader = DataUploader(
                    client=client,
                    dataset_version=dataset_context.dataset_version,
                    datalake=datalake,
                )
                simple_uploader._add_images_to_dataset_version_in_batches(
                    images_to_upload=[
                        os.path.join(dataset_context.images_dir, image_filename)
                        for image_filename in os.listdir(dataset_context.images_dir)
                    ],
                    data_tags=[data_tag],
                )
        else:
            first_annotation = annotations[0]
            if "segmentation" in first_annotation:
                dataset_context.dataset_version.set_type(InferenceType.SEGMENTATION)
            elif "bbox" in first_annotation:
                dataset_context.dataset_version.set_type(InferenceType.OBJECT_DETECTION)
            elif "category_id" in first_annotation:
                dataset_context.dataset_version.set_type(InferenceType.CLASSIFICATION)
            else:
                raise ValueError(
                    f"Unsupported dataset type: {dataset_context.dataset_version.type}"
                )

    data_tags: List[str] = [data_tag]

    if dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
        classification_uploader = ClassificationDatasetContextUploader(
            client=client,
            dataset_context=dataset_context,
            datalake=datalake,
            data_tags=data_tags,
        )
        classification_uploader.upload_dataset_context()

    elif dataset_context.dataset_version.type == InferenceType.OBJECT_DETECTION:
        object_detection_uploader = ObjectDetectionDatasetContextUploader(
            client=client,
            dataset_context=dataset_context,
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
        object_detection_uploader.upload_dataset_context()

    elif dataset_context.dataset_version.type == InferenceType.SEGMENTATION:
        segmentation_uploader = SegmentationDatasetContextUploader(
            client=client,
            dataset_context=dataset_context,
            datalake=datalake,
            data_tags=data_tags,
            use_id=use_id,
            fail_on_asset_not_found=fail_on_asset_not_found,
        )
        segmentation_uploader.upload_dataset_context()
