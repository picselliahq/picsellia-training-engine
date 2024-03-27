import os

import pytest
from picsellia import Client, Dataset, DatasetVersion, Data
from picsellia.types.enums import InferenceType
from picsellia.sdk.asset import MultiAsset


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
    data = datalake.upload_data(filepaths=files)

    return data


@pytest.fixture
def picsellia_client() -> Client:
    try:
        TOKEN = os.environ["PICSELLIA_TEST_TOKEN"]
        HOST = os.environ["PICSELLIA_TEST_HOST"]

    except KeyError as e:
        raise KeyError(
            "FATAL ERROR, you need to define env var PICSELLIA_TEST_TOKEN with api token and PICSELLIA_TEST_HOST with "
            "platform host"
        ) from e

    if HOST == "https://app.picsellia.com":
        raise ValueError("FATAL ERROR, can't test on production")

    return Client(api_token=TOKEN, host=HOST)


@pytest.fixture
def destination_path() -> str:
    return os.path.join(os.getcwd(), "tests", "destination_path")


@pytest.fixture
def dataset(picsellia_client: Client) -> Dataset:
    return picsellia_client.create_dataset("test-picsellia-training-engine")
