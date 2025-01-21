from examples.processing.augmentation.utils.common import process_dataset
from src import pipeline
from src.models.utils.picsellia_context import create_picsellia_processing_context
from src.steps.data_extraction.processing.processing_data_extractor import (
    get_processing_dataset_collection,
)
from src.steps.processing.common.dataset_context_uploader import upload_dataset_context


processing_context = create_picsellia_processing_context(
    processing_parameters={
        "datalake": "default",
        "data_tag": "augmented_data",
    }
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
