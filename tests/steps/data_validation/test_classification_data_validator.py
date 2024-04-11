from typing import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType

from src.steps.data_validation.classification_data_validator import (
    classification_data_validator,
)


class TestDataValidator:
    def test_data_validator(self, mock_dataset_collection: Callable):
        with patch(
            "src.steps.data_validation.utils.classification_dataset_validator.ClassificationDatasetValidator"
            ".validate"
        ) as mocked_validate:
            dataset_collection = mock_dataset_collection(
                dataset_type=InferenceType.CLASSIFICATION
            )
            classification_data_validator.entrypoint(
                dataset_collection=dataset_collection
            )
            assert mocked_validate.call_count == 1
