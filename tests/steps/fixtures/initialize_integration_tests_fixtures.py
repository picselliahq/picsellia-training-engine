import os
import sys

import pytest
from picsellia import Client


def get_images_path(dataset_name: str):
    return os.path.join(
        os.getcwd(),
        "tests",
        "data",
        "classification",
        "Tyre_Quality_Classification",
        f"pytest-{dataset_name}",
        "images",
    )


def get_annotations_path(dataset_name: str):
    return os.path.join(
        os.getcwd(),
        "tests",
        "data",
        "classification",
        "Tyre_Quality_Classification",
        f"pytest-{dataset_name}",
        "annotations",
        "coco.json",
    )


def get_multi_asset(dataset_version):
    return dataset_version.list_assets()


def get_labelmap(dataset_version):
    return {label.name: label for label in dataset_version.list_labels()}


def create_dataset_version(
    dataset, version_name, dataset_type, uploaded_data, annotations_path
):
    dataset_version = dataset.create_version(version_name, type=dataset_type)
    dataset_version.add_data(uploaded_data).wait_for_done()
    dataset_version.import_annotations_coco_file(file_path=annotations_path)

    return dataset_version


def upload_data(picsellia_client, images_path):
    datalake = picsellia_client.get_datalake()

    files = [os.path.join(images_path, file) for file in os.listdir(images_path)]
    data = datalake.upload_data(files)

    return data


@pytest.fixture
def picsellia_client():
    try:
        TOKEN = os.environ["PICSELLIA_TEST_TOKEN"]
        HOST = os.environ["PICSELLIA_TEST_HOST"]
    except KeyError:
        sys.stdout.write(
            "FATAL ERROR, you need to define env var PICSELLIA_TEST_TOKEN with api token and PICSELLIA_TEST_HOST with "
            "platform host"
        )
        sys.exit(1)

    if HOST == "https://app.picsellia.com":
        sys.stdout.write("FATAL ERROR, can't test on production")
        sys.exit(1)
    client = Client(api_token=TOKEN, host=HOST)
    return client


@pytest.fixture
def destination_path():
    return os.path.join(os.getcwd(), "tests", "destination_path")


@pytest.fixture
def dataset(picsellia_client):
    return picsellia_client.create_dataset("test-picsellia-training-engine")
