from src import step
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.processing.processing_paddleocr_data_validator import (
    ProcessingPaddleOcrDataValidator,
)


@step
def paddleocr_processing_data_validator(
    dataset_context: DatasetContext,
) -> None:
    validator = ProcessingPaddleOcrDataValidator(
        dataset_context=dataset_context,
    )
    validator.validate()
