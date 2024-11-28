from typing import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType

from src.steps.data_validation.common.coco_classification_dataset_collection_validator import (
    coco_classification_dataset_collection_validator,
)


class TestDataValidator:
    def test_data_validator(self, mock_dataset_collection: Callable):
        with patch(
            "src.models.steps.data_validation.common.classification_dataset_context_validator"
            ".ClassificationDatasetContextValidator"
            ".validate"
        ) as mocked_validate:
            dataset_collection = mock_dataset_collection(
                dataset_type=InferenceType.CLASSIFICATION
            )
            coco_classification_dataset_collection_validator.entrypoint(
                dataset_collection=dataset_collection
            )
            assert mocked_validate.call_count == 3
