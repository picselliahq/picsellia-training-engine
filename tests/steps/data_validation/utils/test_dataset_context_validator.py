from collections.abc import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType

import pytest

from src.enums import DatasetSplitName
from src.steps.data_validation.utils.image_utils import get_image_path_list
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetContextValidator:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_format(
        self,
        mock_dataset_context: Callable,
        mock_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )
        dataset_context.download_assets()
        dataset_context_validator = mock_dataset_context_validator(
            dataset_context=dataset_context
        )
        with pytest.raises(ValueError):
            dataset_context_validator.validate_image_format(
                image_path_list=["image1.gif", "image2.mov"],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_corruption(
        self,
        mock_dataset_context: Callable,
        mock_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )
        dataset_context.download_assets()
        dataset_context_validator = mock_dataset_context_validator(
            dataset_context=dataset_context
        )
        dataset_context_validator.validate_image_corruption(
            image_path_list=get_image_path_list(
                dataset_context_validator.dataset_context.image_dir
            ),
        )
        with pytest.raises(ValueError):
            dataset_context_validator.validate_image_corruption(
                image_path_list=[
                    "tests/data/corrupted_images/018e75f7-388d-76e6-b3c6-8072b216be04.jpg"
                ],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_extraction(
        self,
        mock_dataset_context: Callable,
        mock_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )
        dataset_context.download_assets()
        dataset_context_validator = mock_dataset_context_validator(
            dataset_context=dataset_context
        )
        dataset_context_validator.validate()
        with pytest.raises(ValueError):
            dataset_context_validator.validate_image_extraction(
                image_path_list=["image1.jpg"],
            )
        with pytest.raises(ValueError):
            dataset_context_validator.dataset_context.multi_asset = [
                "image1.jpg",
                "image2.jpg",
                "image3.jpg",
            ]
            dataset_context_validator.validate_image_extraction(
                image_path_list=[
                    "image1.jpg",
                    "image2.jpg",
                    "image3.jpg",
                    "image4.jpg",
                ],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate(
        self,
        mock_dataset_context: Callable,
        mock_dataset_context_validator: Callable,
        dataset_type: InferenceType,
    ):
        dataset_context = mock_dataset_context(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=dataset_type,
            )
        )
        dataset_context.download_assets()
        dataset_context_validator = mock_dataset_context_validator(
            dataset_context=dataset_context
        )

        with (
            patch.object(
                dataset_context_validator, "validate_image_extraction"
            ) as mock_validate_image_extraction,
            patch.object(
                dataset_context_validator, "validate_image_corruption"
            ) as mock_validate_image_corruption,
            patch.object(
                dataset_context_validator, "validate_image_format"
            ) as mock_validate_image_format,
        ):
            # mock_get_image_path_list.return_value = ["image1.jpg", "image2.jpg"]
            dataset_context_validator.validate()

            assert mock_validate_image_extraction.called
            assert mock_validate_image_corruption.called
            assert mock_validate_image_format.called
