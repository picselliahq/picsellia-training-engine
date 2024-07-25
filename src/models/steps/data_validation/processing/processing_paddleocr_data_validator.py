from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class ProcessingPaddleOcrDataValidator(DatasetContextValidator):
    def __init__(
        self,
        dataset_context: DatasetContext,
    ):
        super().__init__(dataset_context=dataset_context)
