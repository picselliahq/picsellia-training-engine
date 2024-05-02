from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.dataset_context import DatasetContext
from src.models.steps.data_extraction.processing.processing_dataset_context_extractor import (
    ProcessingDatasetContextExtractor,
)


@step
def processing_data_extractor() -> DatasetContext:
    """
    Extracts a dataset from a processing job and prepares it for processing.

    This function retrieves the active processing context from the pipeline, uses it to initialize a
    ProcessingDatasetContextExtractor with the current job and dataset version, and retrieves a DatasetContext
    for the dataset ready for processing. It then downloads all necessary assets and annotations.

    The function is designed to be used as a step in a Picsellia Pipeline, making it part of the automated
    data preparation and processing pipeline.

    Returns:
        - DatasetContext: A dataset context prepared for processing, including all assets and annotations downloaded.

    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    dataset_context_extractor = ProcessingDatasetContextExtractor(
        job_id=context.job_id, dataset_version=context.input_dataset_version
    )
    dataset_context = dataset_context_extractor.get_dataset_context()
    dataset_context.download_assets()
    return dataset_context
