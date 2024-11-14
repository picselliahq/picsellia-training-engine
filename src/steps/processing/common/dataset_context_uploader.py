from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.common.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)
from src.models.steps.processing.common.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)

from picsellia.types.enums import InferenceType

from src.models.steps.processing.common.segmentation_dataset_context_uploader import (
    SegmentationDatasetContextUploader,
)


@step
def dataset_context_uploader(
    dataset_context: DatasetContext,
    use_id: bool = True,
    fail_on_asset_not_found: bool = True,
) -> None:
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    if dataset_context.dataset_version.type == InferenceType.OBJECT_DETECTION:
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
    elif dataset_context.dataset_version.type == InferenceType.CLASSIFICATION:
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
    else:
        raise ValueError(
            f"Unsupported dataset type: {dataset_context.dataset_version.type}"
        )
