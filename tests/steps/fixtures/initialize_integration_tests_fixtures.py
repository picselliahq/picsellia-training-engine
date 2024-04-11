import os

import pytest
from picsellia import Client, Dataset, DatasetVersion, Data
from picsellia.types.enums import InferenceType
from picsellia.sdk.asset import MultiAsset

from src.models.contexts.picsellia_context import PicselliaContext

from picsellia.services.error_manager import ErrorManager


def get_multi_asset(dataset_version: DatasetVersion) -> MultiAsset:
    return dataset_version.list_assets()


def get_labelmap(dataset_version: DatasetVersion) -> dict:
    return {label.name: label for label in dataset_version.list_labels()}


def create_dataset_version(
    dataset: Dataset,
    version_name: str,
    dataset_type: InferenceType,
    uploaded_data: Data,
    annotations_path: str,
) -> DatasetVersion:
    dataset_version = dataset.create_version(version=version_name, type=dataset_type)
    dataset_version.add_data(data=uploaded_data).wait_for_done()
    dataset_version.import_annotations_coco_file(file_path=annotations_path)

    return dataset_version


def upload_data(picsellia_client: Client, images_path: str) -> Data:
    datalake = picsellia_client.get_datalake()

    files = [os.path.join(images_path, file) for file in os.listdir(images_path)]

    error_manager = ErrorManager()
    data = datalake.upload_data(filepaths=files, error_manager=error_manager)
    error_paths = [error.path for error in error_manager.errors]
    while error_paths:
        error_manager = ErrorManager()
        data = datalake.upload_data(filepaths=error_paths, error_manager=error_manager)
        error_paths = [error.path for error in error_manager.errors]

    return data


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
def picsellia_context(api_token: str, host: str) -> PicselliaContext:
    return PicselliaContext(api_token=api_token, host=host)


@pytest.fixture
def destination_path() -> str:
    return os.path.join(os.getcwd(), "tests", "destination_path")


@pytest.fixture
def dataset(picsellia_client: Client) -> Dataset:
    return picsellia_client.create_dataset("test-picsellia-training-engine")
