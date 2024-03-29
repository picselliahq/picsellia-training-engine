import logging
from typing import List, Callable

import pytest
from picsellia import Client, Experiment, Project

from src.steps.data_extraction.utils.dataset_handler import DatasetHandler
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_prop_train_split() -> float:
    return 0.8


@pytest.fixture
def mock_project(
    picsellia_client: Client,
) -> Project:
    project = picsellia_client.create_project("test-picsellia-training-engine")
    return project


@pytest.fixture
def mock_experiment(
    mock_project: Project,
    mock_dataset_version: Callable,
) -> Callable:
    def _mock_experiment(
        experiment_name: str,
        datasets_metadata: List[DatasetTestMetadata],
    ) -> Experiment:
        experiment = mock_project.create_experiment(name=experiment_name)
        for dataset_metadata in datasets_metadata:
            dataset_version = mock_dataset_version(dataset_metadata=dataset_metadata)
            experiment.attach_dataset(
                name=dataset_metadata.attached_name, dataset_version=dataset_version
            )
        return experiment

    return _mock_experiment


@pytest.fixture
def mock_dataset_handler(
    mock_experiment: Callable, mock_prop_train_split: float
) -> Callable:
    def _mock_dataset_handler(
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
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
