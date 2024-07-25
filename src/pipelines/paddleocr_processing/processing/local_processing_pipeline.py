import argparse
import os

from src import pipeline
from src.models.contexts.processing.test_picsellia_processing_context import (
    TestPicselliaProcessingContext,
)
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)
from src.steps.data_validation.processing.processing_paddleocr_data_validator import (
    paddleocr_processing_data_validator,
)
from src.steps.processing.pre_annotation.paddleocr_processing import (
    paddleocr_processing,
)
from src.steps.weights_extraction.processing.paddle_ocr_weights_extractor import (
    paddle_ocr_weights_extractor,
)

parser = argparse.ArgumentParser()
parser.add_argument("--api_token", type=str, help="Picsellia API token")
parser.add_argument("--organization_id", type=str, help="Picsellia organization name")
parser.add_argument("--job_id", type=str, help="Picsellia job id")
parser.add_argument("--input_dataset_version_id", type=str, help="Picsellia dataset id")
parser.add_argument("--model_version_id", type=str, help="Picsellia Model Version id")
args = parser.parse_args()

os.environ["api_token"] = args.api_token
os.environ["organization_id"] = args.organization_id
os.environ["job_id"] = args.job_id
os.environ["input_dataset_version_id"] = args.input_dataset_version_id
os.environ["model_version_id"] = args.model_version_id


def get_context(args):
    return TestPicselliaProcessingContext(
        job_id=args.job_id,
        input_dataset_version_id=args.input_dataset_version_id,
        model_version_id=args.model_version_id,
    )


@pipeline(
    context=get_context(args),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def paddleocr_processing_pipeline() -> None:
    dataset_context = processing_data_extractor()

    paddleocr_processing_data_validator(dataset_context=dataset_context)
    model_collection = paddle_ocr_weights_extractor()
    paddleocr_processing(
        dataset_context=dataset_context, model_collection=model_collection
    )


if __name__ == "__main__":
    paddleocr_processing_pipeline()
