from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)

from src.models.dataset.common.coco_dataset_context import CocoDatasetContext
from src.models.steps.processing.common.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)


@step
def classification_dataset_context_uploader(dataset_context: CocoDatasetContext):
    """
    Uploads a classification dataset context to Picsellia.

    This function retrieves the active processing context from the pipeline and initializes a
    `ClassificationDatasetContextUploader`. It uploads the dataset context (images and annotations)
    to the specified datalake in Picsellia, attaching relevant data tags.

    Args:
        dataset_context (CocoDatasetContext): The dataset context containing the images and annotations
                                          to be uploaded.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    uploader = ClassificationDatasetContextUploader(
        client=context.client,
        dataset_context=dataset_context,
        datalake=context.processing_parameters.datalake,
        data_tags=[
            context.processing_parameters.data_tag,
            dataset_context.dataset_version.version,
        ],
    )
    uploader.upload_dataset_context()
