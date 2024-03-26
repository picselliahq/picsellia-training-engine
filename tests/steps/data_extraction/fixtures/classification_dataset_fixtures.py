import logging
import os
import shutil

import pytest
from picsellia import Client, Dataset, DatasetVersion, Data
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName
from src.steps.data_extraction.utils.dataset_context import DatasetContext
from tests.steps.fixtures.initialize_integration_tests_fixtures import (
    create_dataset_version,
    upload_data,
    get_multi_asset,
    get_labelmap,
)

uploaded_data_registry = []


def get_images_path(
    dataset_split_name: DatasetSplitName, dataset_type: InferenceType
) -> str:
    if dataset_type == InferenceType.CLASSIFICATION:
        return os.path.join(
            os.getcwd(),
            "tests",
            "data",
            "classification",
            "Tyre_Quality_Classification",
            f"pytest-{dataset_split_name.value}",
            "images",
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def get_annotations_path(
    dataset_split_name: DatasetSplitName, dataset_type: InferenceType
) -> str:
    if dataset_type == InferenceType.CLASSIFICATION:
        return os.path.join(
            os.getcwd(),
            "tests",
            "data",
            "classification",
            "Tyre_Quality_Classification",
            f"pytest-{dataset_split_name.value}",
            "annotations",
            "coco.json",
        )
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


@pytest.fixture
def mock_uploaded_data(picsellia_client: Client):
    def _mock_uploaded_data(
        dataset_split_name: DatasetSplitName, dataset_type: InferenceType
    ) -> Data:
        uploaded_data = upload_data(
            picsellia_client=picsellia_client,
            images_path=get_images_path(
                dataset_split_name=dataset_split_name, dataset_type=dataset_type
            ),
        )
        uploaded_data_registry.append(uploaded_data)
        return uploaded_data

    return _mock_uploaded_data


@pytest.fixture
def mock_dataset_version(dataset: Dataset, mock_uploaded_data):
    def _mock_dataset_version(
        dataset_split_name: DatasetSplitName,
        dataset_type: InferenceType,
    ) -> DatasetVersion:
        uploaded_data = mock_uploaded_data(
            dataset_split_name=dataset_split_name,
            dataset_type=dataset_type,
        )
        dataset_version = create_dataset_version(
            dataset=dataset,
            version_name=dataset_split_name.value,
            dataset_type=dataset_type,
            uploaded_data=uploaded_data,
            annotations_path=get_annotations_path(
                dataset_split_name=dataset_split_name, dataset_type=dataset_type
            ),
        )
        return dataset_version

    return _mock_dataset_version


@pytest.fixture
def mock_dataset_context(
    destination_path: str, mock_uploaded_data, mock_dataset_version
):
    def _mock_dataset_context(
        dataset_split_name: DatasetSplitName,
        dataset_type: InferenceType,
    ) -> DatasetContext:
        dataset_version = mock_dataset_version(
            dataset_split_name=dataset_split_name,
            dataset_type=dataset_type,
        )
        dataset_context = DatasetContext(
            name=dataset_split_name.value,
            dataset_version=dataset_version,
            multi_asset=get_multi_asset(dataset_version),
            labelmap=get_labelmap(dataset_version),
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
