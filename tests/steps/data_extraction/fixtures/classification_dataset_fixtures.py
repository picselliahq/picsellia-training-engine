import logging
import shutil

import pytest

from src.steps.data_extraction.utils.dataset_context import DatasetContext
from tests.steps.fixtures.initialize_integration_tests_fixtures import (
    create_dataset_version,
    upload_data,
    get_images_path,
    get_annotations_path,
    get_multi_asset,
    get_labelmap,
)


@pytest.fixture
def mock_train_dataset_context_name():
    return "train"


@pytest.fixture
def mock_train_uploaded_data(picsellia_client, mock_train_dataset_context_name):
    return upload_data(
        picsellia_client, get_images_path(mock_train_dataset_context_name)
    )


@pytest.fixture
def mock_train_dataset_version(
    mock_train_dataset_context_name, dataset, mock_train_uploaded_data
):
    return create_dataset_version(
        dataset,
        mock_train_dataset_context_name,
        "CLASSIFICATION",
        mock_train_uploaded_data,
        get_annotations_path(mock_train_dataset_context_name),
    )


@pytest.fixture
def mock_train_dataset_context(
    mock_train_dataset_context_name, mock_train_dataset_version, destination_path
):
    return DatasetContext(
        name=mock_train_dataset_context_name,
        dataset_version=mock_train_dataset_version,
        multi_asset=get_multi_asset(mock_train_dataset_version),
        labelmap=get_labelmap(mock_train_dataset_version),
        destination_path=destination_path,
    )


@pytest.fixture
def mock_val_dataset_context_name():
    return "val"


@pytest.fixture
def mock_val_uploaded_data(picsellia_client, mock_val_dataset_context_name):
    return upload_data(picsellia_client, get_images_path(mock_val_dataset_context_name))


@pytest.fixture
def mock_val_dataset_version(
    mock_val_dataset_context_name, dataset, mock_val_uploaded_data
):
    return create_dataset_version(
        dataset,
        mock_val_dataset_context_name,
        "CLASSIFICATION",
        mock_val_uploaded_data,
        get_annotations_path(mock_val_dataset_context_name),
    )


@pytest.fixture
def mock_val_dataset_context(
    mock_val_dataset_context_name, mock_val_dataset_version, destination_path
):
    return DatasetContext(
        name=mock_val_dataset_context_name,
        dataset_version=mock_val_dataset_version,
        multi_asset=get_multi_asset(mock_val_dataset_version),
        labelmap=get_labelmap(mock_val_dataset_version),
        destination_path=destination_path,
    )


@pytest.fixture
def mock_test_dataset_context_name():
    return "test"


@pytest.fixture
def mock_test_uploaded_data(picsellia_client, mock_test_dataset_context_name):
    return upload_data(
        picsellia_client, get_images_path(mock_test_dataset_context_name)
    )


@pytest.fixture
def mock_test_dataset_version(
    mock_test_dataset_context_name, dataset, mock_test_uploaded_data
):
    return create_dataset_version(
        dataset,
        mock_test_dataset_context_name,
        "CLASSIFICATION",
        mock_test_uploaded_data,
        get_annotations_path(mock_test_dataset_context_name),
    )


@pytest.fixture
def mock_test_dataset_context(
    mock_test_dataset_context_name, mock_test_dataset_version, destination_path
):
    return DatasetContext(
        name=mock_test_dataset_context_name,
        dataset_version=mock_test_dataset_version,
        multi_asset=get_multi_asset(mock_test_dataset_version),
        labelmap=get_labelmap(mock_test_dataset_version),
        destination_path=destination_path,
    )


@pytest.fixture(autouse=True)
def cleanup(
    mock_train_uploaded_data,
    mock_val_uploaded_data,
    mock_test_uploaded_data,
    dataset,
    destination_path,
):
    def _cleanup():
        try:
            mock_train_uploaded_data.delete()
        except Exception:
            logging.exception(
                "Could not delete data in test datalake. You will need to do it manually."
            )
        try:
            mock_val_uploaded_data.delete()
        except Exception:
            logging.exception(
                "Could not delete data in test datalake. You will need to do it manually."
            )
        try:
            mock_test_uploaded_data.delete()
        except Exception:
            logging.exception(
                "Could not delete data in test datalake. You will need to do it manually."
            )
        try:
            dataset.delete()
        except Exception:
            logging.exception(
                "Could not delete dataset in test. You will need to do it manually."
            )
        try:
            shutil.rmtree(destination_path)
        except Exception:
            logging.exception(
                "Could not delete destination path in test. You will need to do it manually."
            )

    yield
    _cleanup()
