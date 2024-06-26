# type: ignore

from src import pipeline
from src.models.contexts.picsellia_context import (
    PicselliaProcessingContext,
)
from src.models.parameters.processing.processing_slicer_parameters import (
    ProcessingSlicerParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_dataset_collection_extractor,
)
from src.steps.data_validation.processing.processing_slicer_data_validator import (
    slicer_data_validator,
)
from src.steps.processing.dataset_version_creation.dataset_context_uploader import (
    dataset_context_uploader,
)
from src.steps.processing.dataset_version_creation.slicer_processing import (
    slicer_processing,
)


def get_context() -> PicselliaProcessingContext[ProcessingSlicerParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingSlicerParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def slicer_processing_pipeline() -> None:
    dataset_collection = processing_dataset_collection_extractor()
    slicer_data_validator(dataset_context=dataset_collection.input)
    dataset_collection = slicer_processing(dataset_collection=dataset_collection)
    dataset_context_uploader(dataset_context=dataset_collection.output)


if __name__ == "__main__":
    slicer_processing_pipeline()
