from typing import Callable
from unittest.mock import patch

from picsellia.types.enums import InferenceType

from src import Pipeline
from src.enums import DatasetSplitName
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.steps.data_extraction.training.training_data_extractor import (
    training_dataset_collection_extractor,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


class TestTrainingDataExtractor:
    def test_training_data_extractor(self, mock_picsellia_training_context: Callable):
        picsellia_training_context = mock_picsellia_training_context(
            experiment_name="test_experiment",
            datasets_metadata=[
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                )
            ],
            hyperparameters_cls=UltralyticsHyperParameters,
            augmentation_parameters_cls=UltralyticsAugmentationParameters,
            export_parameters_cls=ExportParameters,
        )
        with patch.object(Pipeline, "get_active_context") as mock_get_active_context:
            mock_get_active_context.return_value = picsellia_training_context
            dataset_collection = training_dataset_collection_extractor.entrypoint()
            assert dataset_collection is not None
