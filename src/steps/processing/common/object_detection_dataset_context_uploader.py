from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.common.object_detection_dataset_context_uploader import (
    ObjectDetectionDatasetContextUploader,
)


@step
def object_detection_dataset_context_uploader(dataset_context: DatasetContext):
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    uploader = ObjectDetectionDatasetContextUploader(
        client=context.client,
        dataset_context=dataset_context,
        datalake=context.processing_parameters.datalake,
        data_tags=[
            context.processing_parameters.data_tag,
            dataset_context.dataset_version.version,
        ],
    )
    uploader.upload_dataset_context()
