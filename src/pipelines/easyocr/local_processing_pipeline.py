import argparse
import dataclasses
import os

from src import pipeline
from src.models.contexts.test_picsellia_context import TestPicselliaProcessingContext

from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)

from src.steps.data_validation.processing.processing_easyocr_data_validator import easyocr_data_validator

from src.steps.processing.pre_annotation.easyocr_processing import easyocr_processing

parser = argparse.ArgumentParser()
parser.add_argument("--api_token", type=str, help="Picsellia API token")
parser.add_argument("--organization_id", type=str, help="Picsellia organization name")
parser.add_argument("--job_id", type=str, help="Picsellia job id")
parser.add_argument("--input_dataset_version_id", type=str, help="Picsellia dataset id")
args = parser.parse_args()

os.environ["api_token"] = args.api_token
os.environ["organization_id"] = args.organization_id
os.environ["job_id"] = args.job_id
os.environ["input_dataset_version_id"] = args.input_dataset_version_id


@dataclasses.dataclass
class ProcessingEasyOcrParameters:
    language: str


def get_context(args):
    return TestPicselliaProcessingContext(
        job_id=args.job_id,
        input_dataset_version_id=args.input_dataset_version_id,
        processing_parameters=ProcessingEasyOcrParameters(language="en"),
    )


@pipeline(
    context=get_context(args),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def easyocr_pipeline() -> None:
    dataset_context = processing_data_extractor()

    easyocr_data_validator(dataset_context=dataset_context)

    easyocr_processing(
        dataset_context=dataset_context
    )


if __name__ == "__main__":
    easyocr_pipeline()
