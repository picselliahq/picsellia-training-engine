from typing import Callable, List, Union

import pytest
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import InferenceType

from src.enums import DatasetSplitName
from src.models.dataset.common.dataset_collection import DatasetCollection
from tests.steps.fixtures.dataset_version_fixtures import DatasetTestMetadata


def get_experiment_name(nb_dataset: int) -> str:
    return f"test-{nb_dataset}-dataset" + ("s" if nb_dataset > 1 else "")


def generate_general_params(
    nb_datasets: List[int],
) -> tuple[str, List[DatasetTestMetadata]]:
    for nb_dataset in nb_datasets:
        experiment_name = get_experiment_name(nb_dataset=nb_dataset)
        if nb_dataset == 1:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                )
            ]
        elif nb_dataset == 2:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
            ]
        elif nb_dataset == 3:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
            ]
        else:
            raise ValueError(f"Unsupported number of datasets: {nb_dataset}")
        yield experiment_name, dataset_metadatas


def generate_params_with_split_ratios(
    nb_datasets: List[int],
) -> tuple[str, List[DatasetTestMetadata], Union[List, None], Union[Exception, None]]:
    for nb_dataset in nb_datasets:
        experiment_name = get_experiment_name(nb_dataset=nb_dataset)
        if nb_dataset == 1:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                )
            ]
            split_ratios = [0.8, 0.1, 0.1]
            expected_exception = None
        elif nb_dataset == 2:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
            ]
            split_ratios = [0.8, 0.2]
            expected_exception = None
        elif nb_dataset == 3:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
            ]
            split_ratios = None
            expected_exception = RuntimeError
        else:
            raise ValueError(f"Unsupported number of datasets: {nb_dataset}")
        yield experiment_name, dataset_metadatas, split_ratios, expected_exception


def generate_params_for_nested_datasets(
    nb_datasets: List[int],
) -> tuple[str, List[DatasetTestMetadata], Exception]:
    for nb_dataset in nb_datasets:
        experiment_name = get_experiment_name(nb_dataset=nb_dataset)
        if nb_dataset == 1:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                    attached_name="train_nested",
                )
            ]
            expected_exception = ResourceNotFoundError
            yield experiment_name, dataset_metadatas, expected_exception
        elif nb_dataset == 2:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                    attached_name="test_nested",
                ),
            ]
            expected_exception = ResourceNotFoundError
            yield experiment_name, dataset_metadatas, expected_exception
        elif nb_dataset == 3:
            experiment_name_1 = experiment_name + "_1"
            experiment_name_2 = experiment_name + "_2"
            dataset_metadatas_1 = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                    attached_name="test_nested",
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
            ]
            dataset_metadatas_2 = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                    attached_name="val_nested",
                ),
            ]
            expected_exception = ResourceNotFoundError
            yield experiment_name_1, dataset_metadatas_1, expected_exception
            yield experiment_name_2, dataset_metadatas_2, expected_exception
        elif nb_dataset == 4:
            dataset_metadatas = [
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TRAIN,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.TEST,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                ),
                DatasetTestMetadata(
                    dataset_split_name=DatasetSplitName.VAL,
                    dataset_type=InferenceType.CLASSIFICATION,
                    attached_name="val_nested",
                ),
            ]
            expected_exception = RuntimeError
            yield experiment_name, dataset_metadatas, expected_exception


def get_nb_assets_after_split(nb_dataset: int) -> tuple[int, int, int]:
    if nb_dataset == 1:
        nb_train_assets = 8
        nb_val_assets = 1
        nb_test_assets = 1
    elif nb_dataset == 2:
        nb_train_assets = 8
        nb_val_assets = 2
        nb_test_assets = 10
    elif nb_dataset == 3:
        nb_train_assets = 10
        nb_val_assets = 10
        nb_test_assets = 10
    else:
        raise ValueError(f"Unsupported number of datasets: {nb_dataset}")
    return nb_train_assets, nb_val_assets, nb_test_assets


def check_splited_assets(
    dataset_collection: DatasetCollection, nb_dataset: int
) -> None:
    nb_train_assets, nb_val_assets, nb_test_assets = get_nb_assets_after_split(
        nb_dataset=nb_dataset
    )
    assert (
        len(dataset_collection["train"].assets) == nb_train_assets
    ), f"Incorrect number of train assets for {nb_dataset} dataset(s)"
    assert (
        len(dataset_collection["val"].assets) == nb_val_assets
    ), f"Incorrect number of val assets for {nb_dataset} dataset(s)"
    assert (
        len(dataset_collection["test"].assets) == nb_test_assets
    ), f"Incorrect number of test assets for {nb_dataset} dataset(s)"


def check_multi_asset_in_dataset(dataset_collection):
    assert all(
        asset in dataset_collection["train"].dataset_version.list_assets()
        for asset in dataset_collection["train"].assets
    ), "Train assets mismatch"
    assert all(
        asset in dataset_collection["val"].dataset_version.list_assets()
        for asset in dataset_collection["val"].assets
    ), "Val assets mismatch"
    assert all(
        asset in dataset_collection["test"].dataset_version.list_assets()
        for asset in dataset_collection["test"].assets
    ), "Test assets mismatch"


def check_labelmap(dataset_collection: DatasetCollection, nb_dataset: int) -> None:
    if nb_dataset == 1:
        assert (
            dataset_collection["train"].labelmap
            == dataset_collection["val"].labelmap
            == dataset_collection["test"].labelmap
        ), "Inconsistent labelmap across contexts"
    elif nb_dataset == 2:
        assert (
            dataset_collection["train"].labelmap == dataset_collection["val"].labelmap
        ), "Inconsistent labelmap between train and val"
    elif nb_dataset == 3:
        print("No labelmap check for 3 datasets")


class TestDatasetHandler:
    @pytest.mark.parametrize(
        "experiment_name, datasets",
        list(generate_general_params(nb_datasets=[1, 2, 3])),
    )
    def test_get_dataset_collection(
        self,
        mock_training_dataset_collection_extractor: Callable,
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
    ) -> None:
        dataset_collection_extractor = mock_training_dataset_collection_extractor(
            experiment_name=experiment_name, datasets=datasets
        )
        dataset_collection = dataset_collection_extractor.get_dataset_collection()
        assert dataset_collection["train"].dataset_name == DatasetSplitName.TRAIN.value
        assert dataset_collection["val"].dataset_name == DatasetSplitName.VAL.value
        assert dataset_collection["test"].dataset_name == DatasetSplitName.TEST.value

    @pytest.mark.parametrize(
        "experiment_name, datasets, expected_split_ratios, expected_exception",
        list(generate_params_with_split_ratios(nb_datasets=[1, 2, 3])),
    )
    def test_get_split_ratios(
        self,
        mock_training_dataset_collection_extractor: Callable,
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
        expected_split_ratios: Union[List, None],
        expected_exception: Union[Exception, None],
    ) -> None:
        dataset_collection_extractor = mock_training_dataset_collection_extractor(
            experiment_name=experiment_name, datasets=datasets
        )
        if expected_exception:
            with pytest.raises(expected_exception):
                dataset_collection_extractor._get_split_ratios(
                    nb_attached_datasets=len(datasets)
                )
        else:
            split_ratios = dataset_collection_extractor._get_split_ratios(
                nb_attached_datasets=len(datasets)
            )
            assert split_ratios == expected_split_ratios

    @pytest.mark.parametrize(
        "experiment_name, datasets",
        list(generate_general_params(nb_datasets=[1, 2, 3])),
    )
    def test_handle_datasets(
        self,
        mock_training_dataset_collection_extractor: Callable,
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
    ):
        dataset_collection_extractor = mock_training_dataset_collection_extractor(
            experiment_name=experiment_name, datasets=datasets
        )
        nb_dataset = len(datasets)

        if nb_dataset == 1:
            dataset_collection = dataset_collection_extractor._handle_one_dataset(
                train_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.TRAIN.value
                )
            )
        elif nb_dataset == 2:
            dataset_collection = dataset_collection_extractor._handle_two_datasets(
                train_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.TRAIN.value
                ),
                test_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.TEST.value
                ),
            )
        elif nb_dataset == 3:
            dataset_collection = dataset_collection_extractor._handle_three_datasets(
                train_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.TRAIN.value
                ),
                val_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.VAL.value
                ),
                test_dataset_version=dataset_collection_extractor.experiment.get_dataset(
                    name=DatasetSplitName.TEST.value
                ),
            )
        else:
            pytest.fail(f"Unsupported number of datasets: {nb_dataset}")

        check_splited_assets(
            dataset_collection=dataset_collection, nb_dataset=nb_dataset
        )
        check_labelmap(dataset_collection=dataset_collection, nb_dataset=nb_dataset)
        check_multi_asset_in_dataset(dataset_collection=dataset_collection)

    @pytest.mark.parametrize(
        "experiment_name, datasets, expected_exception",
        list(generate_params_for_nested_datasets(nb_datasets=[1, 2, 3, 4])),
    )
    def test_handle_datasets_exceptions(
        self,
        mock_training_dataset_collection_extractor: Callable,
        experiment_name: str,
        datasets: List[DatasetTestMetadata],
        expected_exception: Exception,
    ):
        with pytest.raises(expected_exception):
            dataset_collection_extractor = mock_training_dataset_collection_extractor(
                experiment_name=experiment_name, datasets=datasets
            )
            dataset_collection_extractor.get_dataset_collection()
