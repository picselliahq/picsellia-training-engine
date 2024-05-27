from typing import Callable

import pytest

from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


@pytest.fixture
def mock_dataset_context_validator() -> Callable:
    def _dataset_context_validator(
        dataset_context: DatasetContext,
    ) -> DatasetContextValidator:
        return DatasetContextValidator(dataset_context=dataset_context)

    return _dataset_context_validator
