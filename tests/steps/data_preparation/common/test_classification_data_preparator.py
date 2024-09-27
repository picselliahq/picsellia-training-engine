from typing import Callable
import tempfile
import os
from picsellia.types.enums import InferenceType
from src.steps.data_preparation.common.classification_data_preparator import (
    classification_data_preparator,
)


class TestClassificationDataPreparator:
    def test_classification_data_preparator(self, mock_dataset_collection: Callable):
        classification_dataset_collection = mock_dataset_collection(
            dataset_type=InferenceType.CLASSIFICATION
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            classification_dataset_collection.download_all(
                destination_path=os.path.join(temp_dir, "dataset"),
                use_id=True,
                skip_asset_listing=False,
            )

            organized_dataset_collection = classification_data_preparator.entrypoint(
                dataset_collection=classification_dataset_collection,
                destination_path=os.path.join(temp_dir, "organized_dataset"),
            )

            assert organized_dataset_collection == classification_dataset_collection
