from typing import Callable

from picsellia.types.enums import InferenceType

from src.steps.data_preparation.common.classification_data_preparator import (
    classification_data_preparator,
)


class TestClassificationDataPreparator:
    def test_classification_data_preparator(self, mock_dataset_collection: Callable):
        classification_dataset_collection = mock_dataset_collection(
            dataset_type=InferenceType.CLASSIFICATION
        )
        classification_dataset_collection.download_assets()
        organized_dataset_collection = classification_data_preparator.entrypoint(
            dataset_collection=classification_dataset_collection
        )
        assert organized_dataset_collection == classification_dataset_collection
