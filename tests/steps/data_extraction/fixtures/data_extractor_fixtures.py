from typing import Callable, Type, List, Dict

import pytest

from src.models.contexts.picsellia_context import PicselliaTrainingContext
from src.models.parameters.augmentation_parameters import AugmentationParameters
from src.models.parameters.hyper_parameters import HyperParameters
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def log_data() -> Dict:
    return {
        "epochs": 3,
        "batch_size": 4,
        "image_size": 256,
    }


@pytest.fixture
def mock_picsellia_training_context(
    api_token: str,
    host: str,
    mock_experiment: Callable,
    log_data: Dict,
) -> Callable:
    def _mock_picsellia_training_context(
        experiment_name: str,
        datasets_metadata: List[DatasetTestMetadata],
        hyperparameters_cls: Type[HyperParameters],
        augmentation_parameters_cls: Type[AugmentationParameters],
    ) -> PicselliaTrainingContext:
        experiment = mock_experiment(
            experiment_name=experiment_name, datasets_metadata=datasets_metadata
        )
        experiment.log_parameters(parameters=log_data)

        return PicselliaTrainingContext(
            hyperparameters_cls=hyperparameters_cls,
            augmentation_parameters_cls=augmentation_parameters_cls,
            api_token=api_token,
            host=host,
            experiment_id=experiment.id,
        )

    return _mock_picsellia_training_context
