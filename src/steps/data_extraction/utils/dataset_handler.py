import os

from picsellia import DatasetVersion, Experiment
from picsellia.exceptions import ResourceNotFoundError

from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from src.steps.data_extraction.utils.dataset_context import DatasetContext
from src.models.dataset.dataset_split_name import DatasetSplitName


def get_labelmap(dataset_version: DatasetVersion) -> dict:
    return {label.name: label for label in dataset_version.list_labels()}


class DatasetHandler:
    def __init__(self, experiment: Experiment, prop_train_split: float):
        self.experiment = experiment
        self.prop_train_split = prop_train_split
        self.destination_path = os.path.join(os.getcwd(), self.experiment.name)

    def get_dataset_collection(self) -> DatasetCollection:
        nb_attached_datasets = len(self.experiment.list_attached_dataset_versions())
        try:
            train_dataset_version = self.experiment.get_dataset(
                DatasetSplitName.TRAIN.value
            )
        except Exception:
            raise ResourceNotFoundError(
                f"Dataset {DatasetSplitName.TRAIN.value} not found in the experiment"
            )

        if nb_attached_datasets == 3:
            try:
                val_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.VAL.value
                )
            except Exception:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.VAL.value} not found"
                )
            try:
                test_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                )
            except Exception:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.TEST.value} not found"
                )
            return self._handle_three_datasets(
                train_dataset_version=train_dataset_version,
                val_dataset_version=val_dataset_version,
                test_dataset_version=test_dataset_version,
            )
        elif nb_attached_datasets == 2:
            try:
                test_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                )
            except Exception:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.TEST.value} not found"
                )
            return self._handle_two_datasets(
                train_dataset_version=train_dataset_version,
                test_dataset_version=test_dataset_version,
            )
        elif nb_attached_datasets == 1:
            return self._handle_one_dataset(train_dataset_version=train_dataset_version)
        else:
            raise RuntimeError(
                "Invalid number of datasets attached to the experiment: "
                "1, 2 or 3 datasets are expected."
            )

    def _handle_three_datasets(
        self,
        train_dataset_version: DatasetVersion,
        val_dataset_version: DatasetVersion,
        test_dataset_version: DatasetVersion,
    ) -> DatasetCollection:
        return DatasetCollection(
            train_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TRAIN.value,
                dataset_version=train_dataset_version,
                multi_asset=train_dataset_version.list_assets(),
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
            val_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.VAL.value,
                dataset_version=val_dataset_version,
                multi_asset=val_dataset_version.list_assets(),
                labelmap=get_labelmap(val_dataset_version),
                destination_path=self.destination_path,
            ),
            test_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TEST.value,
                dataset_version=test_dataset_version,
                multi_asset=test_dataset_version.list_assets(),
                labelmap=get_labelmap(test_dataset_version),
                destination_path=self.destination_path,
            ),
        )

    def _handle_two_datasets(
        self,
        train_dataset_version: DatasetVersion,
        test_dataset_version: DatasetVersion,
    ) -> DatasetCollection:
        split_ratios = self._get_split_ratios(nb_attached_datasets=2)
        split_assets, counts, labels = train_dataset_version.split_into_multi_assets(
            ratios=split_ratios
        )
        train_assets, val_assets = split_assets
        return DatasetCollection(
            train_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TRAIN.value,
                dataset_version=train_dataset_version,
                multi_asset=train_assets,
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
            val_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.VAL.value,
                dataset_version=train_dataset_version,
                multi_asset=val_assets,
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
            test_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TEST.value,
                dataset_version=test_dataset_version,
                multi_asset=test_dataset_version.list_assets(),
                labelmap=get_labelmap(test_dataset_version),
                destination_path=self.destination_path,
            ),
        )

    def _handle_one_dataset(
        self, train_dataset_version: DatasetVersion
    ) -> DatasetCollection:
        split_ratios = self._get_split_ratios(nb_attached_datasets=1)
        split_assets, counts, labels = train_dataset_version.split_into_multi_assets(
            ratios=split_ratios
        )
        train_assets, val_assets, test_assets = split_assets
        return DatasetCollection(
            train_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TRAIN.value,
                dataset_version=train_dataset_version,
                multi_asset=train_assets,
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
            val_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.VAL.value,
                dataset_version=train_dataset_version,
                multi_asset=val_assets,
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
            test_dataset_context=DatasetContext(
                dataset_name=DatasetSplitName.TEST.value,
                dataset_version=train_dataset_version,
                multi_asset=test_assets,
                labelmap=get_labelmap(train_dataset_version),
                destination_path=self.destination_path,
            ),
        )

    def _get_split_ratios(self, nb_attached_datasets: int) -> list:
        if nb_attached_datasets == 1:
            remaining = round((1 - self.prop_train_split), 2)
            val_test_ratio = round(remaining / 2, 2)
            return [
                self.prop_train_split,
                val_test_ratio,
                val_test_ratio,
            ]
        elif nb_attached_datasets == 2:
            return [self.prop_train_split, round(1 - self.prop_train_split, 2)]
        else:
            raise RuntimeError()
