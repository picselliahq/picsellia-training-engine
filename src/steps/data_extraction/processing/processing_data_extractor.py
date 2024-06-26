from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.common.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)
from src.models.steps.data_extraction.processing.processing_dataset_collection_extractor import (
    ProcessingDatasetCollectionExtractor,
)
from src.models.steps.data_extraction.processing.processing_dataset_context_extractor import (
    ProcessingDatasetContextExtractor,
)


@step
def processing_data_extractor(skip_asset_listing: bool = False) -> DatasetContext:
    """
    Extracts a dataset from a processing job and prepares it for processing.

    This function retrieves the active processing context from the pipeline, uses it to initialize a
    ProcessingDatasetContextExtractor with the current job and dataset version, and retrieves a DatasetContext
    for the dataset ready for processing. It then downloads all necessary assets and annotations.

    The function is designed to be used as a step in a Picsellia Pipeline, making it part of the automated
    data preparation and processing pipeline.

    Args:
        skip_asset_listing: Whether to skip listing the dataset's assets.

    Returns:
        A dataset context prepared for processing, including all assets and annotations downloaded.

    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    dataset_context_extractor = ProcessingDatasetContextExtractor(
        job_id=context.job_id,
        dataset_version=context.input_dataset_version,
        use_id=context.use_id,
    )
    dataset_context = dataset_context_extractor.get_dataset_context(
        skip_asset_listing=skip_asset_listing
    )

    if not skip_asset_listing:
        dataset_context.download_assets()

    return dataset_context


@step
def processing_dataset_collection_extractor() -> ProcessingDatasetCollection:
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
    dataset_collection_extractor = ProcessingDatasetCollectionExtractor(
        input_dataset_version=context.input_dataset_version,
        output_dataset_version=context.output_dataset_version,
        job_id=context.job_id,
        use_id=context.use_id,
    )
    dataset_collection = dataset_collection_extractor.get_dataset_collection()
    dataset_collection.download_assets()
    return dataset_collection
