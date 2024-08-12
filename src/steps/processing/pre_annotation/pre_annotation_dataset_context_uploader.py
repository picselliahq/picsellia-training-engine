import logging

from picsellia import Client
from picsellia.types.enums import InferenceType

from src import Pipeline, step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.steps.processing.dataset_version_creation.data_uploader import (
    DataUploader,
)
from src.models.dataset.common.dataset_context import DatasetContext

logger = logging.getLogger("picsellia")


class PreAnnotationDatasetContextUploader(DataUploader):
    def __init__(self, client: Client, dataset_context: DatasetContext):
        super().__init__(client, dataset_context.dataset_version)
        self.client = client
        self.dataset_context = dataset_context

    def upload_dataset_context(self) -> None:
        """
        Uploads the dataset context to Picsellia.
        """
        if self.dataset_context.dataset_version.type != InferenceType.NOT_CONFIGURED:
            self._add_coco_annotations_to_dataset_version(
                annotation_path=self.dataset_context.coco_file_path
            )

        else:
            logger.info(
                f"👉 Since the dataset's type is set to {InferenceType.NOT_CONFIGURED.name}, "
                f"no annotations will be uploaded."
            )


@step
def pre_annotation_dataset_context_uploader(dataset_context: DatasetContext):
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    uploader = PreAnnotationDatasetContextUploader(
        client=context.client,
        dataset_context=dataset_context,
    )
    uploader.upload_dataset_context()
