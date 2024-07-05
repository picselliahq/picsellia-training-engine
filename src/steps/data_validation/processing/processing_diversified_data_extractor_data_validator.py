from src import step
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.processing.processing_diversified_data_extractor_data_validator import (
    ProcessingDiversifiedDataExtractorDataValidator,
)


@step
def diversified_data_extractor_data_validator(
    dataset_context: DatasetContext,
) -> None:
    validator = ProcessingDiversifiedDataExtractorDataValidator(
        dataset_context=dataset_context,
    )
    validator.validate()
