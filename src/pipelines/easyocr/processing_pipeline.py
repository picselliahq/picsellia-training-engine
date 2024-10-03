from src import pipeline
from src.models.contexts.picsellia_context import PicselliaProcessingContext

from src.models.parameters.processing.processing_easyorc_parameters import ProcessingEasyOcrParameters
from src.steps.data_extraction.processing.processing_data_extractor import (
    processing_data_extractor,
)

from src.steps.data_validation.processing.processing_easyocr_data_validator import easyocr_data_validator

from src.steps.processing.pre_annotation.easyocr_processing import easyocr_processing


def get_context() -> (
        PicselliaProcessingContext[ProcessingEasyOcrParameters]
):
    return PicselliaProcessingContext(
        processing_parameters_cls=ProcessingEasyOcrParameters,
    )


@pipeline(
    context=get_context(),
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
