from src import step
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.processing.processing_easyocr_data_validator import (
    ProcessingEasyOcrDataValidator,
)


@step
def easyocr_data_validator(
    dataset_context: DatasetContext,
) -> None:
    validator = ProcessingEasyOcrDataValidator(
        dataset_context=dataset_context,
    )
    validator.validate()
