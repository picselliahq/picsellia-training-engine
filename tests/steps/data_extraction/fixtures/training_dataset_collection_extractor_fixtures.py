import logging
from typing import List, Callable
from uuid import uuid4

import pytest
from picsellia import Client, Experiment, Project

from src.models.steps.data_extraction.training.training_dataset_collection_extractor import (
    TrainingDatasetCollectionExtractor,
)
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


@pytest.fixture
def mock_train_set_split_ratio() -> float:
    return 0.8


@pytest.fixture
def mock_project(
    picsellia_client: Client,
) -> Project:
    return picsellia_client.create_project(f"test-picsellia-training-engine-{uuid4()}")


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
def mock_experiment_dataset_collection_extractor(
    mock_experiment: Callable, mock_train_set_split_ratio: float
) -> Callable:
    def _mock_dataset_handler(
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
    ) -> TrainingDatasetCollectionExtractor:
        experiment = mock_experiment(experiment_name, datasets)
        return TrainingDatasetCollectionExtractor(
            experiment, mock_train_set_split_ratio
        )

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
