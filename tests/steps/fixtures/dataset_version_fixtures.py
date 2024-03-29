import logging
import os
import shutil
from typing import Callable, Optional

import pytest
from picsellia import Client, Dataset, DatasetVersion, Data
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from tests.steps.fixtures.initialize_integration_tests_fixtures import (
    create_dataset_version,
    upload_data,
)


uploaded_data_registry = []


class DatasetTestMetadata:
    def __init__(
        self,
        dataset_split_name: DatasetSplitName,
        dataset_type: InferenceType,
        attached_name: Optional[str] = None,
    ):
        self.dataset_split_name = dataset_split_name
        self.dataset_type = dataset_type
        self.attached_name = attached_name or dataset_split_name.value


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
            version_name=dataset_metadata.attached_name,
            dataset_type=dataset_metadata.dataset_type,
            uploaded_data=uploaded_data,
            annotations_path=get_annotations_path(dataset_metadata=dataset_metadata),
        )
        return dataset_version

    return _mock_dataset_version


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
