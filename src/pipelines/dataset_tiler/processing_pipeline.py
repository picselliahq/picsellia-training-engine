# type: ignore

from src import pipeline
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.parameters.processing.processing_tiler_parameters import (
    ProcessingTilerParameters,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_dataset_collection_extractor,
)
from src.steps.data_validation.processing.processing_tiler_data_validator import (
    tiler_data_validator,
)
from src.steps.processing.common.dataset_context_uploader import (
    dataset_context_uploader,
)
from src.steps.processing.dataset_version_creation.tiler_processing import (
    tiler_processing,
)


def get_context() -> PicselliaProcessingContext[ProcessingTilerParameters]:
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingTilerParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def tiler_processing_pipeline() -> None:
    dataset_collection = processing_dataset_collection_extractor()
    tiler_data_validator(dataset_context=dataset_collection.input)
    output_dataset_context = tiler_processing(dataset_collection=dataset_collection)
    dataset_context_uploader(dataset_context=output_dataset_context)


if __name__ == "__main__":
    tiler_processing_pipeline()
