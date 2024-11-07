import os
from typing import Optional

from src import Pipeline
from src import step
from src.models.contexts.processing.picsellia_processing_context import (
    PicselliaProcessingContext,
)
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import DatasetContext


def get_destination_path(job_id: Optional[str]) -> str:
    """
    Generates a destination path based on the current working directory and a job ID.

    Args:
        job_id (Optional[str]): The ID of the current job. If None, defaults to "current_job".

    Returns:
        str: The generated file path for the job.
    """
    if not job_id:
        return os.path.join(os.getcwd(), "current_job")
    return os.path.join(os.getcwd(), str(job_id))


@step
def processing_dataset_context_extractor(
    skip_asset_listing: bool = False,
) -> DatasetContext:
    """
    Extracts a dataset context from a processing job, preparing it for further processing.

    This function retrieves the active processing context from the pipeline, uses the input dataset version,
    and creates a `DatasetContext`. The dataset context includes all necessary assets (e.g., images) and
    annotations (e.g., COCO format) required for processing. It downloads the assets and annotations into a
    destination folder based on the current job ID.

    Args:
        skip_asset_listing (bool): Whether to skip listing the dataset's assets during the download process. Defaults to False.

    Returns:
        DatasetContext: The dataset context prepared for processing, including all downloaded assets and annotations.
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
    Extracts a dataset collection from a processing job, preparing it for further processing.

    This function retrieves the active processing context from the pipeline, initializes a
    `ProcessingDatasetCollectionExtractor` with the input and output dataset versions, and downloads
    the necessary assets and annotations for both input and output datasets. It prepares the dataset collection
    and stores them in a specified destination folder.

    Args:
        skip_asset_listing (bool): Whether to skip listing the dataset's assets during the download process. Defaults to False.

    Returns:
        DatasetCollection: The dataset collection prepared for processing, including all downloaded assets and annotations.
    """
    context: PicselliaProcessingContext = Pipeline.get_active_context()
    input_dataset_context = DatasetContext(
        dataset_name="input",
        dataset_version=context.input_dataset_version,
        assets=context.input_dataset_version.list_assets(),
        labelmap=None,
    )
    output_dataset_context = DatasetContext(
        dataset_name="output",
        dataset_version=context.output_dataset_version,
        assets=None,
        labelmap=None,
    )
    dataset_collection = DatasetCollection(
        [input_dataset_context, output_dataset_context]
    )
    dataset_collection.download_all(
        destination_path=get_destination_path(context.job_id),
        use_id=True,
        skip_asset_listing=skip_asset_listing,
    )
    return dataset_collection
