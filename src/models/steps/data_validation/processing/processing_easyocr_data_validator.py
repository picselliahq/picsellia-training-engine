from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.object_detection_dataset_context_validator import (
    ObjectDetectionDatasetContextValidator,
)


class ProcessingEasyOcrDataValidator(ObjectDetectionDatasetContextValidator):
    def __init__(
        self,
        dataset_context: DatasetContext,
    ):
        super().__init__(dataset_context=dataset_context)
