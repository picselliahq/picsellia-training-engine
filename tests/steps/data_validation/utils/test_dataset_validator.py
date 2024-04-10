from collections.abc import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType

import pytest

from src.steps.data_validation.utils.dataset_validator import get_image_path_list


class TestDatasetValidator:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_format(
        self, mock_dataset_validator: Callable, dataset_type: InferenceType
    ):
        dataset_validator = mock_dataset_validator(dataset_type=dataset_type)
        with pytest.raises(ValueError):
            dataset_validator.validate_image_format(
                dataset_context=dataset_validator.dataset_collection.train,
                image_path_list=["image1.gif", "image2.mov"],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_corruption(
        self, mock_dataset_validator: Callable, dataset_type: InferenceType
    ):
        dataset_validator = mock_dataset_validator(dataset_type=dataset_type)
        dataset_validator.validate_image_corruption(
            dataset_context=dataset_validator.dataset_collection.train,
            image_path_list=get_image_path_list(
                dataset_validator.dataset_collection.train.image_dir
            ),
        )
        with pytest.raises(ValueError):
            dataset_validator.validate_image_corruption(
                dataset_context=dataset_validator.dataset_collection.train,
                image_path_list=[
                    "tests/data/corrupted_images/018e75f7-388d-76e6-b3c6-8072b216be04.jpg"
                ],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_image_extraction(
        self, mock_dataset_validator: Callable, dataset_type: InferenceType
    ):
        dataset_validator = mock_dataset_validator(dataset_type=dataset_type)
        with pytest.raises(ValueError):
            dataset_validator.validate_image_extraction(
                dataset_context=dataset_validator.dataset_collection.train,
                image_path_list=["image1.jpg"],
            )
        with pytest.raises(ValueError):
            dataset_validator.dataset_collection.train.multi_asset = [
                "image1.jpg",
                "image2.jpg",
                "image3.jpg",
            ]
            dataset_validator.validate_image_extraction(
                dataset_context=dataset_validator.dataset_collection.train,
                image_path_list=[
                    "image1.jpg",
                    "image2.jpg",
                    "image3.jpg",
                    "image4.jpg",
                ],
            )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_common(
        self, mock_dataset_validator: Callable, dataset_type: InferenceType
    ):
        dataset_validator = mock_dataset_validator(dataset_type=dataset_type)
        dataset_validator._validate_common()

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate(
        self, mock_dataset_validator: Callable, dataset_type: InferenceType
    ):
        dataset_validator = mock_dataset_validator(dataset_type=dataset_type)
        dataset_validator.validate()
        dataset_validator.validate()

        with (
            patch(
                "src.steps.data_validation.utils.dataset_validator.get_image_path_list"
            ) as mock_get_image_path_list,
            patch.object(
                dataset_validator, "validate_image_extraction"
            ) as mock_validate_image_extraction,
            patch.object(
                dataset_validator, "validate_image_corruption"
            ) as mock_validate_image_corruption,
            patch.object(
                dataset_validator, "validate_image_format"
            ) as mock_validate_image_format,
        ):
            mock_get_image_path_list.return_value = ["image1.jpg", "image2.jpg"]
            dataset_validator._validate_common()

            assert mock_get_image_path_list.called
            assert mock_validate_image_extraction.called
            assert mock_validate_image_corruption.called
            assert mock_validate_image_format.called
