import os
from typing import Callable

from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestDatasetOrganizer:
    def test_extract_categories(self, mock_classification_dataset_organizer: Callable):
        classification_dataset_organizer = mock_classification_dataset_organizer(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )

        extracted_categories_dict = (
            classification_dataset_organizer._extract_categories()
        )
        dataset_categories_list = (
            classification_dataset_organizer.dataset_context.labelmap.keys()
        )
        assert len(extracted_categories_dict) == len(dataset_categories_list)
        for category_id, category_name in extracted_categories_dict.items():
            assert category_name in dataset_categories_list

    def test_organizer_creates_category_directories(
        self,
        mock_classification_dataset_organizer: Callable,
    ):
        classification_dataset_organizer = mock_classification_dataset_organizer(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )
        classification_dataset_organizer.organize()

        for (
            category
        ) in classification_dataset_organizer.dataset_context.labelmap.keys():
            category_dir = os.path.join(
                classification_dataset_organizer.dataset_context.image_dir, category
            )
            print(f"category_dir: {category_dir}")
            assert os.path.isdir(category_dir)

    def test_organizer_copies_images_to_correct_directories(
        self,
        mock_classification_dataset_organizer: Callable,
    ):
        classification_dataset_organizer = mock_classification_dataset_organizer(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )
        classification_dataset_organizer.organize()
        for image in classification_dataset_organizer.dataset_context.coco_file.images:
            category_id = next(
                ann.category_id
                for ann in classification_dataset_organizer.dataset_context.coco_file.annotations
                if ann.image_id == image.id
            )
            category_name = next(
                cat.name
                for cat in classification_dataset_organizer.dataset_context.coco_file.categories
                if cat.id == category_id
            )
            expected_path = os.path.join(
                classification_dataset_organizer.dataset_context.image_dir,
                category_name,
                image.file_name,
            )
            assert os.path.exists(
                expected_path
            ), f"Image {image.file_name} should have been copied to {expected_path}."

    def test_cleanup_removes_original_images_dir(
        self,
        mock_classification_dataset_organizer: Callable,
        destination_path: str,
    ):
        classification_dataset_organizer = mock_classification_dataset_organizer(
            dataset_metadata=DatasetTestMetadata(
                dataset_split_name=DatasetSplitName.TRAIN,
                dataset_type=InferenceType.CLASSIFICATION,
            )
        )
        original_images_dir = os.path.join(destination_path, "images")
        os.makedirs(original_images_dir, exist_ok=True)
        classification_dataset_organizer.dataset_context.image_dir = str(
            original_images_dir
        )

        classification_dataset_organizer._cleanup()

        assert not os.path.exists(
            original_images_dir
        ), "The original images directory should be removed after cleanup."
