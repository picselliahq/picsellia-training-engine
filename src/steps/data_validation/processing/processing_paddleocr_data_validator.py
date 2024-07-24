from src import Pipeline, step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.parameters.processing.processing_easyorc_parameters import ProcessingEasyOcrParameters
from src.models.steps.data_validation.processing.processing_easyocr_data_validator import ProcessingEasyOcrDataValidator
from src.models.steps.data_validation.processing.processing_paddleocr_data_validator import \
    ProcessingPaddleOcrDataValidator


@step
def paddleocr_processing_data_validator(
        dataset_context: DatasetContext,
) -> None:
    validator = ProcessingPaddleOcrDataValidator(
        dataset_context=dataset_context,
    )
    validator.validate()
