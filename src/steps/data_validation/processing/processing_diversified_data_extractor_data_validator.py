from src import step
from src.models.dataset.common.base_dataset_context import TBaseDatasetContext
from src.models.steps.data_validation.processing.processing_diversified_data_extractor_data_validator import (
    ProcessingDiversifiedDataExtractorDataValidator,
)


@step
def validate_diversified_data_extractor_data(
    dataset_context: TBaseDatasetContext,
) -> None:
    validator = ProcessingDiversifiedDataExtractorDataValidator(
        dataset_context=dataset_context,
    )
    validator.validate()
