from typing import Callable

import pytest

from src.models.dataset.common.coco_dataset_context import CocoDatasetContext
from src.models.steps.data_validation.common.coco_classification_dataset_context_validator import (
    CocoClassificationDatasetContextValidator,
)


@pytest.fixture
def mock_classification_dataset_context_validator() -> Callable:
    def _classification_dataset_context_validator(
        dataset_context: CocoDatasetContext,
    ) -> CocoClassificationDatasetContextValidator:
        return CocoClassificationDatasetContextValidator(
            dataset_context=dataset_context
        )

    return _classification_dataset_context_validator
