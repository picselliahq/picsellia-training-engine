import os
from typing import Optional

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.processing.processing_dataset_collection import (
    ProcessingDatasetCollection,
)

from src.models.steps.data_extraction.processing.processing_dataset_collection_extractor import (
    ProcessingDatasetCollectionExtractor,
)
from src.models.steps.data_extraction.processing.processing_dataset_context_extractor import (
    ProcessingDatasetContextExtractor,
)


def get_destination_path(job_id: Optional[str]) -> str:
    if not job_id:
        return os.path.join(os.getcwd(), "current_job")
    return os.path.join(os.getcwd(), str(job_id))


@step
def processing_dataset_context_extractor(
    skip_asset_listing: bool = False,
) -> DatasetContext:
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
        dataset_version=context.input_dataset_version,
        use_id=context.use_id,
    )
    destination_path = get_destination_path(context.job_id)
    dataset_context = dataset_context_extractor.get_dataset_context(
        destination_path=destination_path, skip_asset_listing=skip_asset_listing
    )

    if not skip_asset_listing:
        dataset_context.download_assets(image_dir=dataset_context.image_dir)
        dataset_context.download_and_build_coco_file(
            annotations_dir=dataset_context.annotations_dir
        )

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
        use_id=context.use_id,
        download_annotations=context.download_annotations,
    )
    destination_path = get_destination_path(context.job_id)
    dataset_collection = dataset_collection_extractor.get_dataset_collection(
        destination_path=destination_path
    )
    dataset_collection.download_all()
    return dataset_collection
