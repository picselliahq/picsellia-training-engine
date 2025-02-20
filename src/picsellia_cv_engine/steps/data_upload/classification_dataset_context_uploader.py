from src.picsellia_cv_engine import Pipeline
from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)

from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_upload.classification_dataset_context_uploader import (
    ClassificationDatasetContextUploader,
)


@step
def upload_classification_dataset_context(dataset_context: CocoDatasetContext):
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
