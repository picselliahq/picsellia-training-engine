from unittest.mock import patch

import pytest

from src.steps.data_validation.utils.classification_dataset_validator import (
    ClassificationDatasetValidator,
)


class TestClassificationDatasetValidator:
    def test_validate_labelmap(
        self, classification_dataset_validator: ClassificationDatasetValidator
    ):
        classification_dataset_validator.dataset_collection.train.labelmap = {
            "class1": 0
        }
        with pytest.raises(ValueError):
            classification_dataset_validator.validate_labelmap()

    def test_validate_at_least_one_image_per_class(
        self, classification_dataset_validator: ClassificationDatasetValidator
    ):
        classification_dataset_validator.dataset_collection.train.labelmap = {
            "class1": 0,
            "class2": 1,
        }
        with pytest.raises(ValueError):
            classification_dataset_validator.validate_at_least_one_image_per_class()

    def test_validate(
        self, classification_dataset_validator: ClassificationDatasetValidator
    ):
        with patch.object(
            classification_dataset_validator, "_validate_common"
        ) as mock_validate_common, patch.object(
            classification_dataset_validator, "validate_labelmap"
        ) as mock_validate_labelmap, patch.object(
            classification_dataset_validator, "validate_at_least_one_image_per_class"
        ) as mock_validate_at_least_one:
            classification_dataset_validator.validate()

            mock_validate_common.assert_called_once()
            mock_validate_labelmap.assert_called_once()
            mock_validate_at_least_one.assert_called_once()
