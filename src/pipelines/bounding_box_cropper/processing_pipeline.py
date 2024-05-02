# type: ignore

from src import pipeline
from src.models.contexts.picsellia_context import (
    PicselliaProcessingContext,
)
from src.models.parameters.processing.bounding_box_cropper_parameters import (
    BoundingBoxCropperParameters,
)
from src.steps.data_extraction.data_extractor import processing_data_extractor
from src.steps.data_validation.processing.bounding_box_cropper_data_validator import (
    bounding_box_cropper_data_validator,
)
from src.steps.processing.bounding_box_cropper_processing import (
    bounding_box_cropper_processing,
)


def get_context() -> PicselliaProcessingContext[BoundingBoxCropperParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=BoundingBoxCropperParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def bounding_box_cropper_processing_pipeline() -> None:
    dataset_context = processing_data_extractor()
    bounding_box_cropper_data_validator(dataset_context=dataset_context)
    bounding_box_cropper_processing(dataset_context=dataset_context)


if __name__ == "__main__":
    bounding_box_cropper_processing_pipeline()
