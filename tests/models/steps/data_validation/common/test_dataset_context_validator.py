import tempfile
from collections.abc import Callable
from unittest.mock import patch, Mock
import pytest
from picsellia.types.enums import InferenceType
from src.enums import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetContextValidator:
    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_images_format(
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

        # Use a temporary directory as the destination path
        with tempfile.TemporaryDirectory() as destination_path:
            dataset_context.download_assets(destination_path=destination_path)

            dataset_context_validator = mock_dataset_context_validator(
                dataset_context=dataset_context
            )

            # Test with invalid image formats
            with pytest.raises(ValueError):
                dataset_context_validator.validate_images_format(
                    images_path_list=["image1.gif", "image2.mov"],
                )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_images_corruption(
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

        # Use a temporary directory as the destination path
        with tempfile.TemporaryDirectory() as destination_path:
            dataset_context.download_assets(destination_path=destination_path)

            dataset_context_validator = mock_dataset_context_validator(
                dataset_context=dataset_context
            )

            # Simulate valid images
            with patch("PIL.Image.open") as mock_open:
                mock_img = Mock()
                mock_open.return_value.__enter__.return_value = mock_img
                dataset_context_validator.validate_images_corruption(
                    images_path_list=["image1.jpg", "image2.jpg"]
                )
                mock_open.assert_called()

            # Test with a corrupted image
            with pytest.raises(ValueError), patch(
                "PIL.Image.open", side_effect=Exception
            ):
                dataset_context_validator.validate_images_corruption(
                    images_path_list=[
                        "tests/data/corrupted_images/image_corrupted.jpg"
                    ],
                )

    @pytest.mark.parametrize("dataset_type", [InferenceType.CLASSIFICATION])
    def test_validate_images_extraction(
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

        # Use a temporary directory as the destination path
        with tempfile.TemporaryDirectory() as destination_path:
            dataset_context.download_assets(destination_path=destination_path)

            dataset_context_validator = mock_dataset_context_validator(
                dataset_context=dataset_context
            )

            # Test with fewer images than assets
            with pytest.raises(ValueError):
                dataset_context_validator.validate_images_extraction(
                    images_path_list=["image1.jpg"],
                )

            # Test with more images than assets
            with pytest.raises(ValueError):
                dataset_context_validator.validate_images_extraction(
                    images_path_list=[
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

        # Use a temporary directory as the destination path
        with tempfile.TemporaryDirectory() as destination_path:
            dataset_context.download_assets(destination_path=destination_path)

            dataset_context_validator = mock_dataset_context_validator(
                dataset_context=dataset_context
            )

            # Mock the internal validation methods to ensure they are called
            with (
                patch.object(
                    dataset_context_validator, "validate_images_extraction"
                ) as mock_validate_images_extraction,
                patch.object(
                    dataset_context_validator, "validate_images_corruption"
                ) as mock_validate_images_corruption,
                patch.object(
                    dataset_context_validator, "validate_images_format"
                ) as mock_validate_images_format,
            ):
                dataset_context_validator.validate()

                # Ensure each validation method is called during the validation process
                assert mock_validate_images_extraction.called
                assert mock_validate_images_corruption.called
                assert mock_validate_images_format.called
