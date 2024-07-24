from src import pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext

from src.models.parameters.processing.processing_paddleocr_parameters import ProcessingPaddleOcrParameters
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)
from src.steps.data_validation.processing.processing_paddleocr_data_validator import paddleocr_processing_data_validator
from src.steps.processing.pre_annotation.paddleocr_processing import paddleocr_processing


def get_context() -> (
        PicselliaProcessingContext[ProcessingPaddleOcrParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingPaddleOcrParameters,
    )


@pipeline(
    context=get_context(),
    log_folder_path="logs/",
    remove_logs_on_completion=False,
)
def paddleocr_processing_pipeline() -> None:
    dataset_context = processing_data_extractor()

    paddleocr_processing_data_validator(dataset_context=dataset_context)

    paddleocr_processing(
        dataset_context=dataset_context
    )


if __name__ == "__main__":
    paddleocr_processing_pipeline()
