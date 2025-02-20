from examples.processing.augmentation.utils.common import process_dataset
from src import pipeline
from src.models.utils.local_context import create_local_processing_context
from src.steps.data_extraction.processing.processing_data_extractor import (
    get_processing_dataset_collection,
)
from src.steps.processing.common.dataset_context_uploader import upload_dataset_context

from picsellia.types.enums import ProcessingType
import argparse

parser = argparse.ArgumentParser("Local Augmentations Pipeline")
parser.add_argument("--api_token", type=str, required=True)
parser.add_argument("--organization_id", type=str, required=True)
parser.add_argument("--results_dir", type=str, required=True)
parser.add_argument("--job_type", type=str, required=True)
parser.add_argument("--input_dataset_version_id", type=str, required=True)
parser.add_argument("--output_dataset_version_name", type=str, required=True)


args = parser.parse_args()

processing_context = create_local_processing_context(
    api_token=args.api_token,
    organization_id=args.organization_id,
    job_id=args.results_dir,
    job_type=ProcessingType(args.job_type),
    input_dataset_version_id=args.input_dataset_version_id,
    output_dataset_version_name=args.output_dataset_version_name,
    processing_parameters={
        "datalake": "default",
        "data_tag": "augmented_data",
        "num_augmentations": "4",
    },
)


@pipeline(
    context=processing_context,
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def augmentations_pipeline():
    dataset_collection = get_processing_dataset_collection()
    dataset_collection["output"] = process_dataset(
        dataset_collection["input"], dataset_collection["output"]
    )
    upload_dataset_context(dataset_collection["output"], use_id=False)
    return dataset_collection


if __name__ == "__main__":
    augmentations_pipeline()
