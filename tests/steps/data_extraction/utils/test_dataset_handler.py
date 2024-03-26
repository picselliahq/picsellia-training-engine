import os
from typing import List

import pytest
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import InferenceType

from src.models.dataset.dataset_split_name import DatasetSplitName


def generate_params(
    num_datasets: List[int], with_ratios=False, with_num_datasets=False
):
    ratios_exceptions = {
        1: ([0.8, 0.1, 0.1], None),
        2: ([0.8, 0.2], None),
        3: (None, RuntimeError),
    }

    dataset_combinations = [
        (i, [DatasetSplitName.TRAIN, DatasetSplitName.TEST, DatasetSplitName.VAL][:i])
        for i in range(1, 4)
    ]

    for i, dataset_splits in dataset_combinations:
        if i in num_datasets:
            datasets = [
                (split, InferenceType.CLASSIFICATION) for split in dataset_splits
            ]
            name = f"test-{i}-dataset" + ("s" if i > 1 else "")
            if with_ratios:
                expected_ratios, expected_exception = ratios_exceptions[i]
                yield name, datasets, expected_ratios, expected_exception
            elif with_num_datasets:
                yield name, datasets, i
            else:
                yield name, datasets


def generate_random_dataset_name():
    return "random_dataset_" + os.urandom(4).hex()


def check_splited_assets(dataset_collection, num_datasets):
    expected_counts = {
        1: (8, 1, 1),
        2: (8, 2, 10),
        3: (10, 10, 10),
    }
    train, val, test = expected_counts[num_datasets]
    assert (
        len(dataset_collection.train.multi_asset) == train
    ), f"Incorrect number of train assets for {num_datasets} dataset(s)"
    assert (
        len(dataset_collection.val.multi_asset) == val
    ), f"Incorrect number of val assets for {num_datasets} dataset(s)"
    assert (
        len(dataset_collection.test.multi_asset) == test
    ), f"Incorrect number of test assets for {num_datasets} dataset(s)"


def check_multi_asset(dataset_collection):
    assert all(
        asset in dataset_collection.train.dataset_version.list_assets()
        for asset in dataset_collection.train.multi_asset
    ), "Train assets mismatch"
    assert all(
        asset in dataset_collection.val.dataset_version.list_assets()
        for asset in dataset_collection.val.multi_asset
    ), "Val assets mismatch"
    assert all(
        asset in dataset_collection.test.dataset_version.list_assets()
        for asset in dataset_collection.test.multi_asset
    ), "Test assets mismatch"


def check_labelmap(dataset_collection, num_datasets):
    if num_datasets == 1:
        assert (
            dataset_collection.train.labelmap
            == dataset_collection.val.labelmap
            == dataset_collection.test.labelmap
        ), "Inconsistent labelmap across contexts"
    elif num_datasets == 2:
        assert (
            dataset_collection.train.labelmap == dataset_collection.val.labelmap
        ), "Inconsistent labelmap between train and val"
    elif num_datasets == 3:
        print("No labelmap check for 3 datasets")


class TestDatasetHandler:
    @pytest.mark.parametrize(
        "experiment_name, datasets",
        list(generate_params(num_datasets=[1, 2, 3])),
    )
    def test_get_dataset_collection(
        self, mock_dataset_handler, experiment_name, datasets
    ):
        dataset_handler = mock_dataset_handler(experiment_name, datasets)
        dataset_collection = dataset_handler.get_dataset_collection()
        assert dataset_collection.train.name == DatasetSplitName.TRAIN.value
        assert dataset_collection.val.name == DatasetSplitName.VAL.value
        assert dataset_collection.test.name == DatasetSplitName.TEST.value

    @pytest.mark.parametrize(
        "experiment_name, datasets, expected_ratios, expected_exception",
        list(generate_params(num_datasets=[1, 2, 3], with_ratios=True)),
    )
    def test_get_split_ratios(
        self,
        mock_dataset_handler,
        experiment_name,
        datasets,
        expected_ratios,
        expected_exception,
    ):
        dataset_handler = mock_dataset_handler(experiment_name, datasets)
        if expected_exception:
            with pytest.raises(expected_exception):
                dataset_handler._get_split_ratios(count_datasets=len(datasets))
        else:
            split_ratios = dataset_handler._get_split_ratios(
                count_datasets=len(datasets)
            )
            assert split_ratios == expected_ratios

    @pytest.mark.parametrize(
        "experiment_name, datasets, num_datasets",
        list(generate_params(num_datasets=[1, 2, 3], with_num_datasets=True)),
    )
    def test_handle_datasets(
        self, mock_dataset_handler, experiment_name, datasets, num_datasets
    ):
        dataset_handler = mock_dataset_handler(experiment_name, datasets)

        if num_datasets == 1:
            dataset_collection = dataset_handler._handle_one_dataset(
                train_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.TRAIN.value
                )
            )
        elif num_datasets == 2:
            dataset_collection = dataset_handler._handle_two_datasets(
                train_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.TRAIN.value
                ),
                test_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                ),
            )
        elif num_datasets == 3:
            dataset_collection = dataset_handler._handle_three_datasets(
                train_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.TRAIN.value
                ),
                val_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.VAL.value
                ),
                test_dataset_version=dataset_handler.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                ),
            )
        else:
            pytest.fail(f"Unsupported number of datasets: {num_datasets}")

        check_splited_assets(dataset_collection, num_datasets)

        check_labelmap(dataset_collection, num_datasets)

        check_multi_asset(dataset_collection)

    @pytest.mark.parametrize(
        "experiment_name, datasets, expected_exception",
        [
            (
                "test-one-random-dataset",
                [
                    (
                        DatasetSplitName.TRAIN,
                        InferenceType.CLASSIFICATION,
                        generate_random_dataset_name(),
                    )
                ],
                ResourceNotFoundError,
            ),
            (
                "test-two-random-datasets",
                [
                    (DatasetSplitName.TRAIN, InferenceType.CLASSIFICATION),
                    (
                        DatasetSplitName.TEST,
                        InferenceType.CLASSIFICATION,
                        generate_random_dataset_name(),
                    ),
                ],
                ResourceNotFoundError,
            ),
            (
                "test-three-random-datasets",
                [
                    (DatasetSplitName.TRAIN, InferenceType.CLASSIFICATION),
                    (
                        DatasetSplitName.TEST,
                        InferenceType.CLASSIFICATION,
                        generate_random_dataset_name(),
                    ),
                    (
                        DatasetSplitName.VAL,
                        InferenceType.CLASSIFICATION,
                        generate_random_dataset_name(),
                    ),
                ],
                ResourceNotFoundError,
            ),
        ],
    )
    def test_handle_datasets_exceptions(
        self, mock_dataset_handler, experiment_name, datasets, expected_exception
    ):
        with pytest.raises(expected_exception):
            dataset_handler = mock_dataset_handler(experiment_name, datasets)
            dataset_handler.get_dataset_collection()
