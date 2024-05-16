import os
from typing import Dict, List
from uuid import uuid4

import pytest
from picsellia import Client, Data, Datalake, Dataset, DatasetVersion, Label
from picsellia.sdk.asset import MultiAsset
from picsellia.services.error_manager import ErrorManager
from picsellia.types.enums import InferenceType

from src.models.contexts.common.picsellia_context import PicselliaContext


def get_multi_asset(dataset_version: DatasetVersion) -> MultiAsset:
    return dataset_version.list_assets()


def get_labelmap(dataset_version: DatasetVersion) -> Dict[str, Label]:
    return {label.name: label for label in dataset_version.list_labels()}


def create_dataset_version(
    dataset: Dataset,
    version_name: str,
    dataset_type: InferenceType,
    uploaded_data: List[Data],
    annotations_path: str,
) -> DatasetVersion:
    dataset_version = dataset.create_version(version=version_name, type=dataset_type)
    job = dataset_version.add_data(data=uploaded_data)
    job.wait_for_done(attempts=50)
    dataset_version.import_annotations_coco_file(file_path=annotations_path)

    return dataset_version


def upload_data(picsellia_client: Client, images_path: str) -> List[Data]:
    datalake = picsellia_client.get_datalake()

    files = [os.path.join(images_path, file) for file in os.listdir(images_path)]

    all_uploaded_data = []

    error_manager = ErrorManager()
    data = datalake.upload_data(
        filepaths=files, error_manager=error_manager, tags=["pytest"]
    )
    if isinstance(data, Data):
        data = [data]
    all_uploaded_data.extend([one_data for one_data in data])
    error_paths = [error.path for error in error_manager.errors]
    while error_paths:
        error_manager = ErrorManager()
        data = datalake.upload_data(
            filepaths=error_paths, error_manager=error_manager, tags=["pytest"]
        )
        if isinstance(data, Data):
            data = [data]
        all_uploaded_data.extend([one_data for one_data in data])
        error_paths = [error.path for error in error_manager.errors]

    return all_uploaded_data


@pytest.fixture
def api_token() -> str:
    try:
        return os.environ["PICSELLIA_TEST_TOKEN"]
    except KeyError as e:
        raise KeyError(
            "FATAL ERROR, you need to define env var PICSELLIA_TEST_TOKEN with api token"
        ) from e


@pytest.fixture
def host() -> str:
    try:
        HOST = os.environ["PICSELLIA_TEST_HOST"]
        if HOST == "https://app.picsellia.com":
            raise ValueError("FATAL ERROR, can't test on production")
        return HOST
    except KeyError as e:
        raise KeyError(
            "FATAL ERROR, you need to define env var PICSELLIA_TEST_HOST with platform host"
        ) from e


@pytest.fixture
def picsellia_client(api_token: str, host: str) -> Client:
    return Client(api_token=api_token, host=host)


@pytest.fixture
def picsellia_default_datalake(api_token: str, host: str) -> Datalake:
    return Client(api_token=api_token, host=host).get_datalake()


@pytest.fixture
def picsellia_context(api_token: str, host: str) -> PicselliaContext:
    return PicselliaContext(api_token=api_token, host=host)


@pytest.fixture
def destination_path() -> str:
    return os.path.join(os.getcwd(), "tests", "destination_path")


@pytest.fixture
def dataset(picsellia_client: Client) -> Dataset:
    return picsellia_client.create_dataset(f"test-training-engine-{str(uuid4())[:-10]}")
