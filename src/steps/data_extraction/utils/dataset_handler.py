import os
from typing import List

from picsellia import DatasetVersion, Experiment, Label
from picsellia.exceptions import ResourceNotFoundError

from src.models.dataset.dataset_collection import DatasetCollection
from src.models.dataset.dataset_context import DatasetContext
from src.enums import DatasetSplitName


def get_labelmap(dataset_version: DatasetVersion) -> dict[str, Label]:
    """
    Retrieves the label map from a dataset version.

    Parameters:
        dataset_version (DatasetVersion): The dataset version from which to retrieve the label map.

    Returns:
        dict: A dictionary mapping label names to label objects.
    """
    return {label.name: label for label in dataset_version.list_labels()}


class DatasetHandler:
    """
    Manages dataset versions attached to an experiment and prepares dataset collections for processing.

    This class provides functionality to retrieve dataset versions from an experiment,
    organize them into dataset contexts based on training, validation, and testing splits,
    and assemble these contexts into a DatasetCollection for convenient access and use.

    Attributes:
        experiment (Experiment): The experiment from which datasets are to be retrieved.
        train_set_split_ratio (float): The proportion of the dataset to be used for training when only one dataset is attached.
        destination_path (str): The local path where datasets will be stored and accessed.
    """

    def __init__(self, experiment: Experiment, train_set_split_ratio: float):
        """
        Initializes a DatasetHandler with an experiment and configuration for dataset splits.

        Args:
            experiment (Experiment): The Picsellia Experiment object.
            train_set_split_ratio (float): The proportion of data to allocate to the training split.
        """
        self.experiment = experiment
        self.train_set_split_ratio = train_set_split_ratio
        self.destination_path = os.path.join(os.getcwd(), self.experiment.name)

    def get_dataset_collection(self) -> DatasetCollection:
        """
        Retrieves dataset versions attached to the experiment and organizes them into a DatasetCollection.

        This method handles different scenarios based on the number of attached datasets: one, two, or three.
        It prepares dataset contexts for each scenario and assembles them into a DatasetCollection.

        Returns:
            - DatasetCollection: A collection of dataset contexts prepared based on the attached dataset versions.

        Raises:
            - ResourceNotFoundError: If the expected dataset splits are not found in the experiment.
            - RuntimeError: If an invalid number of datasets are attached to the experiment.

        Returns:
            - DatasetCollection: A collection of dataset contexts prepared based on the attached dataset versions.
        """
        nb_attached_datasets = len(self.experiment.list_attached_dataset_versions())
        try:
            train_dataset_version = self.experiment.get_dataset(
                DatasetSplitName.TRAIN.value
            )
        except Exception as e:
            raise ResourceNotFoundError(
                "Training dataset not found in the experiment."
            ) from e

        if nb_attached_datasets == 3:
            try:
                val_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.VAL.value
                )
            except Exception as e:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.VAL.value} not found."
                ) from e
            try:
                test_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                )
            except Exception as e:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.TEST.value} not found"
                ) from e
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
            except Exception as e:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but {DatasetSplitName.TEST.value} not found"
                ) from e
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
        """
        Handles the scenario where three distinct datasets (train, validation, and test) are attached to the experiment.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version for the training split.
            val_dataset_version (DatasetVersion): The dataset version for the validation split.
            test_dataset_version (DatasetVersion): The dataset version for the test split.

        Returns:
            DatasetCollection: A collection with distinct contexts for training, validation, and testing splits.
        """
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
        """
        Handles the scenario where two datasets are attached to the experiment, requiring a split of the first for training and validation.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version used for both training and validation splits.
            test_dataset_version (DatasetVersion): The dataset version for the test split.

        Returns:
            DatasetCollection: A collection with contexts for training, validation, and testing splits, with the first dataset split for the first two.
        """
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
        """
        Handles the scenario where a single dataset is attached to the experiment, requiring splitting into training, validation, and test splits.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version to be split into training, validation, and test contexts.

        Returns:
            DatasetCollection: A collection with contexts for training, validation, and testing splits, all derived from the single dataset version.
        """
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

    def _get_split_ratios(self, nb_attached_datasets: int) -> List[float]:
        """
        Determines the split ratios for dividing a single dataset into training, validation, and testing splits based on the configuration.

        Parameters:
            nb_attached_datasets (int): The number of datasets attached to the experiment.

        Returns:
            List[float]: A list of split ratios for training, validation, and testing splits.

        Raises:
            RuntimeError: If an invalid number of attached datasets is provided.
        """
        if nb_attached_datasets == 1:
            remaining = round((1 - self.train_set_split_ratio), 2)
            val_test_ratio = round(remaining / 2, 2)
            return [
                self.train_set_split_ratio,
                val_test_ratio,
                val_test_ratio,
            ]
        elif nb_attached_datasets == 2:
            return [
                self.train_set_split_ratio,
                round(1 - self.train_set_split_ratio, 2),
            ]
        else:
            raise RuntimeError(
                "Invalid number of datasets attached to the experiment: "
                "1, 2 or 3 datasets are expected."
            )
