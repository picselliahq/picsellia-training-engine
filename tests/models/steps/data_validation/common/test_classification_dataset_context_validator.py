from typing import Callable

import pytest
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestClassificationDatasetValidator:
    def test_validate_labelmap(
        self,
        mock_dataset_context: Callable,
        mock_classification_dataset_context_validator: Callable,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )
        classification_dataset_context_validator = (
            mock_classification_dataset_context_validator(
                dataset_context=dataset_context
            )
        )
        classification_dataset_context_validator.dataset_context.labelmap = {
            "class1": 0
        }
        with pytest.raises(ValueError):
            classification_dataset_context_validator.validate_labelmap()

    def test_validate_at_least_one_image_per_class(
        self,
        mock_dataset_context: Callable,
        mock_classification_dataset_context_validator: Callable,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )
        classification_dataset_context_validator = (
            mock_classification_dataset_context_validator(
                dataset_context=dataset_context
            )
        )
        classification_dataset_context_validator.dataset_context.labelmap = {
            "class1": 0,
            "class2": 1,
        }
        with pytest.raises(ValueError):
            classification_dataset_context_validator.validate_coco_file()
