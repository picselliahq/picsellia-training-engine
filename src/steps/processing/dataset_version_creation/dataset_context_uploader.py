from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.processing.dataset_version_creation.dataset_context_uploader import (
    DatasetContextUploader,
)


@step
def dataset_context_uploader(dataset_context: DatasetContext):
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    uploader = DatasetContextUploader(
        client=context.client,
        dataset_context=dataset_context,
        datalake=context.processing_parameters.datalake,
        images_tags=[
            context.processing_parameters.images_tag,
            dataset_context.dataset_version.version,
            "picsellia_sliced",
        ],
    )
    uploader.upload_dataset_context()
