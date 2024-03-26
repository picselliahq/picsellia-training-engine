import logging
from typing import List, Tuple, Optional

import pytest
from picsellia import Client, Experiment
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from src.steps.data_extraction.utils.dataset_handler import DatasetHandler


@pytest.fixture
def mock_prop_train_split():
    return 0.8


@pytest.fixture
def mock_project(
    picsellia_client: Client,
):
    project = picsellia_client.create_project("test-picsellia-training-engine")
    return project


@pytest.fixture
def mock_experiment(
    mock_project,
    mock_dataset_version,
):
    def _mock_experiment(
        experiment_name: str,
        datasets: (
            List[Tuple[DatasetSplitName, InferenceType]]
            | List[Tuple[DatasetSplitName, InferenceType, str]]
        ),
    ) -> Experiment:
        experiment = mock_project.create_experiment(name=experiment_name)
        for dataset_info in datasets:
            dataset_split_name, dataset_type = dataset_info[:2]
            attached_name = dataset_info[2] if len(dataset_info) > 2 else None

            dataset_version = mock_dataset_version(dataset_split_name, dataset_type)

            name_to_attach = (
                attached_name if attached_name else dataset_split_name.value
            )
            experiment.attach_dataset(
                name=name_to_attach, dataset_version=dataset_version
            )

        return experiment

    return _mock_experiment


@pytest.fixture
def mock_dataset_handler(mock_experiment, mock_prop_train_split: float):
    def _mock_dataset_handler(
        experiment_name: str,
        datasets: List[Tuple[DatasetSplitName, InferenceType, Optional[str]]],
    ) -> DatasetHandler:
        experiment = mock_experiment(experiment_name, datasets)
        return DatasetHandler(experiment, mock_prop_train_split)

    return _mock_dataset_handler


@pytest.fixture(autouse=True)
def clean_up(mock_project):
    def _clean_up():
        try:
            mock_project.delete()
        except Exception:
            logging.exception("Couldn't delete the project. Please delete it manually")

    yield

    _clean_up()
