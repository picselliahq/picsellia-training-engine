from typing import Callable
import pytest
from unittest.mock import patch
from picsellia.types.enums import InferenceType
from src.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestClassificationDatasetValidator:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_labelmap(
        self,
        mock_dataset_context: Callable,
        mock_classification_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )

        # Validate the labelmap with only one class (should raise an error)
        classification_dataset_context_validator = (
            mock_classification_dataset_context_validator(
                dataset_context=dataset_context
            )
        )
        classification_dataset_context_validator.dataset_context.labelmap = {
            "class1": 0  # Only one class
        }

        # Expect a ValueError because a valid classification dataset must have at least 2 classes
        with pytest.raises(ValueError):
            classification_dataset_context_validator._validate_labelmap()

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_at_least_one_image_per_class(
        self,
        mock_dataset_context: Callable,
        mock_classification_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )

        # Define a mock destination path
        with patch(
            "src.models.dataset.common.dataset_context.DatasetContext.load_coco_file_data"
        ) as mock_load_coco_data:
            # Simulating COCO file content with no images for one class
            mock_load_coco_data.return_value = {
                "categories": [
                    {"id": 1, "name": "class1"},
                    {"id": 2, "name": "class2"},
                ],
                "annotations": [
                    {"category_id": 1},  # Images for class1 only
                ],
            }

            classification_dataset_context_validator = (
                mock_classification_dataset_context_validator(
                    dataset_context=dataset_context
                )
            )
            classification_dataset_context_validator.dataset_context.labelmap = {
                "class1": 0,
                "class2": 1,  # Simulating the labelmap with two classes
            }

            # Expect a ValueError because there are no images for class2
            with pytest.raises(ValueError):
                classification_dataset_context_validator._validate_coco_file()
