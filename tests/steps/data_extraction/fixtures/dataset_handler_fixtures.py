import logging

import pytest

from src.steps.data_extraction.utils.dataset_handler import DatasetHandler


@pytest.fixture
def mock_prop_train_split():
    return 0.8


@pytest.fixture
def mock_project(
    picsellia_client,
    mock_train_dataset_version,
    mock_val_dataset_version,
    mock_test_dataset_version,
):
    project = picsellia_client.create_project("test-picsellia-training-engine")
    project.attach_dataset(dataset_version=mock_train_dataset_version)
    project.attach_dataset(dataset_version=mock_val_dataset_version)
    project.attach_dataset(dataset_version=mock_test_dataset_version)
    return project


@pytest.fixture
def mock_experiment_one_dataset(mock_project, mock_train_dataset_version):
    experiment = mock_project.create_experiment("test-one-dataset")
    experiment.attach_dataset(name="train", dataset_version=mock_train_dataset_version)
    return experiment


@pytest.fixture
def mock_experiment_two_datasets(
    mock_project, mock_train_dataset_version, mock_test_dataset_version
):
    experiment = mock_project.create_experiment("test-two-datasets")
    experiment.attach_dataset(name="train", dataset_version=mock_train_dataset_version)
    experiment.attach_dataset(name="test", dataset_version=mock_test_dataset_version)
    return experiment


@pytest.fixture
def mock_experiment_three_datasets(
    mock_project,
    mock_train_dataset_version,
    mock_val_dataset_version,
    mock_test_dataset_version,
):
    experiment = mock_project.create_experiment("test-three-datasets")
    experiment.attach_dataset(name="train", dataset_version=mock_train_dataset_version)
    experiment.attach_dataset(name="val", dataset_version=mock_val_dataset_version)
    experiment.attach_dataset(name="test", dataset_version=mock_test_dataset_version)
    return experiment


@pytest.fixture
def mock_dataset_handler_one_dataset(
    mock_experiment_one_dataset, mock_prop_train_split
):
    return DatasetHandler(mock_experiment_one_dataset, mock_prop_train_split)


@pytest.fixture
def mock_dataset_handler_two_datasets(
    mock_experiment_two_datasets, mock_prop_train_split
):
    return DatasetHandler(mock_experiment_two_datasets, mock_prop_train_split)


@pytest.fixture
def mock_dataset_handler_three_datasets(
    mock_experiment_three_datasets, mock_prop_train_split
):
    return DatasetHandler(mock_experiment_three_datasets, mock_prop_train_split)


@pytest.fixture(autouse=True)
def clean_up(mock_project):
    def _clean_up():
        try:
            mock_project.delete()
        except Exception:
            logging.exception("Couldn't delete the project. Please delete it manually")

    yield

    _clean_up()
