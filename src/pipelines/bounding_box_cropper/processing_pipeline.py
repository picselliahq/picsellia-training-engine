# type: ignore

from src import pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.parameters.processing.processing_bounding_box_cropper_parameters import (
    ProcessingBoundingBoxCropperParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_dataset_collection_extractor,
)
from src.steps.data_validation.processing.processing_bounding_box_cropper_data_validator import (
    bounding_box_cropper_data_validator,
)
from src.steps.processing.dataset_version_creation.bounding_box_cropper_processing import (
    bounding_box_cropper_processing,
)
from src.steps.processing.common.classification_dataset_context_uploader import (
    classification_dataset_context_uploader,
)


def get_context() -> PicselliaProcessingContext[ProcessingBoundingBoxCropperParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingBoundingBoxCropperParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_collection = processing_dataset_collection_extractor()
    bounding_box_cropper_data_validator(dataset_context=dataset_collection["input"])
    output_dataset_context = bounding_box_cropper_processing(
        dataset_collection=dataset_collection
    )
    classification_dataset_context_uploader(dataset_context=output_dataset_context)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()
