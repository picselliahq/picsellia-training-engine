import os
from typing import Optional

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import DatasetContext

from src.models.steps.data_extraction.processing.processing_dataset_collection_extractor import (
    ProcessingDatasetCollectionExtractor,
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
    dataset_context = DatasetContext(
        dataset_name="input",
        dataset_version=context.input_dataset_version,
        assets=context.input_dataset_version.list_assets(),
        labelmap=None,
    )
    destination_path = get_destination_path(context.job_id)

    dataset_context.download_assets(
        destination_path=os.path.join(
            destination_path, dataset_context.dataset_name, "images"
        ),
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )
    dataset_context.download_and_build_coco_file(
        destination_path=os.path.join(
            destination_path, dataset_context.dataset_name, "annotations"
        ),
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )

    return dataset_context


@step
def processing_dataset_collection_extractor(
    skip_asset_listing: bool = False,
) -> DatasetCollection:
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
    )
    destination_path = get_destination_path(context.job_id)
    dataset_collection = dataset_collection_extractor.get_dataset_collection()
    dataset_collection.download_all(
        destination_path=destination_path,
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )
    return dataset_collection
