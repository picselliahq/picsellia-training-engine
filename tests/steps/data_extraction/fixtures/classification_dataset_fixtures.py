import logging
import os
import shutil
from typing import Callable

import pytest
from picsellia import Client, Dataset, DatasetVersion, Data
from picsellia.types.enums import InferenceType

from src.steps.data_extraction.utils.dataset_context import DatasetContext
from tests.steps.data_extraction.utils.conftest import DatasetTestMetadata
from tests.steps.fixtures.initialize_integration_tests_fixtures import (
    create_dataset_version,
    upload_data,
    get_multi_asset,
    get_labelmap,
)

uploaded_data_registry = []


def get_dataset_name(dataset_type: InferenceType) -> str:
    if dataset_type == InferenceType.CLASSIFICATION:
        return "Tyre_Quality_Classification"
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def get_data_path(
    dataset_metadata: DatasetTestMetadata,
) -> str:
    return os.path.join(
        os.getcwd(),
        "tests",
        "data",
        dataset_metadata.dataset_type.value.lower(),
        get_dataset_name(dataset_type=dataset_metadata.dataset_type),
        f"pytest-{dataset_metadata.dataset_split_name.value}",
    )


def get_images_path(
    dataset_metadata: DatasetTestMetadata,
) -> str:
    return os.path.join(get_data_path(dataset_metadata=dataset_metadata), "images")


def get_annotations_path(
    dataset_metadata: DatasetTestMetadata,
) -> str:
    return os.path.join(
        get_data_path(dataset_metadata=dataset_metadata), "annotations", "coco.json"
    )


@pytest.fixture
def mock_uploaded_data(picsellia_client: Client) -> Callable:
    def _mock_uploaded_data(dataset_metadata: DatasetTestMetadata) -> Data:
        uploaded_data = upload_data(
            picsellia_client=picsellia_client,
            images_path=get_images_path(dataset_metadata=dataset_metadata),
        )
        uploaded_data_registry.append(uploaded_data)
        return uploaded_data

    return _mock_uploaded_data


@pytest.fixture
def mock_dataset_version(dataset: Dataset, mock_uploaded_data: Callable) -> Callable:
    def _mock_dataset_version(dataset_metadata: DatasetTestMetadata) -> DatasetVersion:
        uploaded_data = mock_uploaded_data(dataset_metadata=dataset_metadata)
        dataset_version = create_dataset_version(
            dataset=dataset,
            version_name=dataset_metadata.dataset_split_name.value,
            dataset_type=dataset_metadata.dataset_type,
            uploaded_data=uploaded_data,
            annotations_path=get_annotations_path(dataset_metadata=dataset_metadata),
        )
        return dataset_version

    return _mock_dataset_version


@pytest.fixture
def mock_dataset_context(
    destination_path: str, mock_uploaded_data: Callable, mock_dataset_version: Callable
) -> Callable:
    def _mock_dataset_context(dataset_metadata: DatasetTestMetadata) -> DatasetContext:
        dataset_version = mock_dataset_version(dataset_metadata=dataset_metadata)
        dataset_context = DatasetContext(
            dataset_name=dataset_metadata.dataset_split_name.value,
            dataset_version=dataset_version,
            multi_asset=get_multi_asset(dataset_version=dataset_version),
            labelmap=get_labelmap(dataset_version=dataset_version),
            destination_path=destination_path,
        )
        return dataset_context

    return _mock_dataset_context


@pytest.fixture(autouse=True)
def cleanup(
    dataset: Dataset,
    destination_path: str,
):
    def _cleanup():
        for data in uploaded_data_registry:
            try:
                data.delete()
            except Exception:
                logging.exception(
                    "Could not delete data in test. You will need to do it manually."
                )
        try:
            dataset.delete()
        except Exception:
            logging.exception(
                "Could not delete dataset in test. You will need to do it manually."
            )
        try:
            if os.path.exists(destination_path):
                shutil.rmtree(destination_path)
        except Exception:
            logging.exception(
                "Could not delete destination path in test. You will need to do it manually."
            )

    yield
    _cleanup()
