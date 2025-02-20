from typing import List, Dict, Type
from picsellia import DatasetVersion, Experiment
from picsellia.exceptions import ResourceNotFoundError
from picsellia.types.enums import LogType

from src.picsellia_cv_engine.enums import DatasetSplitName
from src.picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)

import logging

logger = logging.getLogger(__name__)


class TrainingDatasetCollectionExtractor:
    """
    Manages dataset versions attached to an experiment and prepares dataset collections for processing.

    This class provides functionality to retrieve dataset versions from an experiment,
    organize them into dataset contexts based on training, validation, and testing splits,
    and assemble these contexts into a DatasetCollection for convenient access and use.

    Attributes:
        experiment (Experiment): The experiment from which datasets are to be retrieved.
        train_set_split_ratio (float): The proportion of the dataset to be used for training when only one dataset is attached.
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

    def get_dataset_collection(
        self, context_class: Type[TBaseDatasetContext], random_seed=None
    ) -> DatasetCollection[TBaseDatasetContext]:
        """
        Retrieves dataset versions attached to the experiment and organizes them into a DatasetCollection.

        This method handles different scenarios based on the number of attached datasets: one, two, or three.
        It prepares dataset contexts for each scenario and assembles them into a DatasetCollection.

        Returns:
            - DatasetCollection: A collection of dataset contexts prepared based on the attached dataset versions.

        Raises:
            - ResourceNotFoundError: If the expected dataset splits are not found in the experiment.
            - RuntimeError: If an invalid number of datasets are attached to the experiment.
        """
        nb_attached_datasets = len(self.experiment.list_attached_dataset_versions())
        try:
            train_dataset_version = self.experiment.get_dataset(
                DatasetSplitName.TRAIN.value
            )
        except Exception as e:
            raise ResourceNotFoundError(
                f"A training dataset is required and couldn't be found. "
                f"Please attach a dataset with the alias '{DatasetSplitName.TRAIN.value}'."
            ) from e

        if nb_attached_datasets == 3:
            try:
                val_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.VAL.value
                )
            except Exception as e:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but a validation dataset couldn't be found. "
                    f"Please attach a dataset with the alias '{DatasetSplitName.VAL.value}'."
                ) from e
            try:
                test_dataset_version = self.experiment.get_dataset(
                    DatasetSplitName.TEST.value
                )
            except Exception as e:
                raise ResourceNotFoundError(
                    f"Found {nb_attached_datasets} attached datasets but a test dataset couldn't be found. "
                    f"Please attach a dataset with the alias '{DatasetSplitName.TEST.value}'."
                ) from e
            return self._handle_three_datasets(
                context_class=context_class,
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
                    f"Found {nb_attached_datasets} attached datasets but a test dataset couldn't be found. "
                    f"Please attach a dataset with the alias '{DatasetSplitName.TEST.value}'."
                ) from e
            return self._handle_two_datasets(
                context_class=context_class,
                train_dataset_version=train_dataset_version,
                test_dataset_version=test_dataset_version,
                random_seed=random_seed,
            )
        elif nb_attached_datasets == 1:
            return self._handle_one_dataset(
                context_class=context_class,
                train_dataset_version=train_dataset_version,
                random_seed=random_seed,
            )
        else:
            raise RuntimeError(
                "Invalid number of datasets attached to the experiment: "
                "1, 2 or 3 datasets are expected."
            )

    def _handle_three_datasets(
        self,
        context_class: Type[TBaseDatasetContext],
        train_dataset_version: DatasetVersion,
        val_dataset_version: DatasetVersion,
        test_dataset_version: DatasetVersion,
    ) -> DatasetCollection[TBaseDatasetContext]:
        """
        Handles the scenario where three distinct datasets (train, validation, and test) are attached to the experiment.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version for the training split.
            val_dataset_version (DatasetVersion): The dataset version for the validation split.
            test_dataset_version (DatasetVersion): The dataset version for the test split.

        Returns:
            DatasetCollection: A collection with distinct contexts for training, validation, and testing splits.
        """
        self.log_distribution(
            train_dataset_version.retrieve_stats().label_repartition,
            "train/objects_distribution",
        )
        self.log_distribution(
            val_dataset_version.retrieve_stats().label_repartition,
            "val/objects_distribution",
        )
        self.log_distribution(
            test_dataset_version.retrieve_stats().label_repartition,
            "test/objects_distribution",
        )

        return DatasetCollection(
            [
                context_class(
                    dataset_name=DatasetSplitName.TRAIN.value,
                    dataset_version=train_dataset_version,
                    assets=train_dataset_version.list_assets(),
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.VAL.value,
                    dataset_version=val_dataset_version,
                    assets=val_dataset_version.list_assets(),
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.TEST.value,
                    dataset_version=test_dataset_version,
                    assets=test_dataset_version.list_assets(),
                    labelmap=None,
                ),
            ],
        )

    def _handle_two_datasets(
        self,
        context_class: Type[TBaseDatasetContext],
        train_dataset_version: DatasetVersion,
        test_dataset_version: DatasetVersion,
        random_seed=None,
    ) -> DatasetCollection[TBaseDatasetContext]:
        """
        Handles the scenario where two datasets are attached to the experiment, requiring a split of the first for training and validation.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version used for both training and validation splits.
            test_dataset_version (DatasetVersion): The dataset version for the test split.

        Returns:
            DatasetCollection: A collection with contexts for training, validation, and testing splits, with the first dataset split for the first two.
        """
        split_ratios = self._get_split_ratios(nb_attached_datasets=2)
        (
            split_assets,
            distributions,
            labels,
        ) = train_dataset_version.split_into_multi_assets(
            ratios=split_ratios, random_seed=random_seed
        )
        train_assets, val_assets = split_assets

        train_distribution, val_distribution = distributions
        self.log_distribution(train_distribution, "train/objects_distribution")
        self.log_distribution(val_distribution, "val/objects_distribution")
        self.log_distribution(
            test_dataset_version.retrieve_stats().label_repartition,
            "test/objects_distribution",
        )

        return DatasetCollection(
            [
                context_class(
                    dataset_name=DatasetSplitName.TRAIN.value,
                    dataset_version=train_dataset_version,
                    assets=train_assets,
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.VAL.value,
                    dataset_version=train_dataset_version,
                    assets=val_assets,
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.TEST.value,
                    dataset_version=test_dataset_version,
                    assets=test_dataset_version.list_assets(),
                    labelmap=None,
                ),
            ]
        )

    def _handle_one_dataset(
        self,
        context_class: Type[TBaseDatasetContext],
        train_dataset_version: DatasetVersion,
        random_seed=None,
    ) -> DatasetCollection[TBaseDatasetContext]:
        """
        Handles the scenario where a single dataset is attached to the experiment, requiring splitting into training, validation, and test splits.

        Parameters:
            train_dataset_version (DatasetVersion): The dataset version to be split into training, validation, and test contexts.

        Returns:
            DatasetCollection: A collection with contexts for training, validation, and testing splits, all derived from the single dataset version.
        """
        split_ratios = self._get_split_ratios(nb_attached_datasets=1)
        (
            split_assets,
            distributions,
            labels,
        ) = train_dataset_version.split_into_multi_assets(
            ratios=split_ratios, random_seed=random_seed
        )
        train_assets, val_assets, test_assets = split_assets

        train_distribution, val_distribution, test_distribution = distributions
        self.log_distribution(train_distribution, "train/objects_distribution")
        self.log_distribution(val_distribution, "val/objects_distribution")
        self.log_distribution(test_distribution, "test/objects_distribution")

        return DatasetCollection(
            [
                context_class(
                    dataset_name=DatasetSplitName.TRAIN.value,
                    dataset_version=train_dataset_version,
                    assets=train_assets,
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.VAL.value,
                    dataset_version=train_dataset_version,
                    assets=val_assets,
                    labelmap=None,
                ),
                context_class(
                    dataset_name=DatasetSplitName.TEST.value,
                    dataset_version=train_dataset_version,
                    assets=test_assets,
                    labelmap=None,
                ),
            ]
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

    def log_distribution(self, distribution: Dict[str, int], log_name: str) -> None:
        """
        Logs the label distribution into the experiment, updating it if it already exists.

        Args:
            distribution (dict): The distribution to log.
            log_name (str): The name of the log.
        """
        distribution_to_log = {
            "x": list(distribution.keys()),
            "y": list(distribution.values()),
        }

        try:
            picsellia_distribution = self.experiment.get_log(name=log_name)
            picsellia_distribution.update(data=distribution_to_log)
            logger.info(f"Updated existing log for {log_name}.")
        except ResourceNotFoundError:
            try:
                self.experiment.log(
                    name=log_name,
                    data=distribution_to_log,
                    type=LogType.BAR,
                    replace=True,
                )
                logger.info(f"Logged new distribution for {log_name}: {distribution}")
            except Exception as e:
                logger.error(f"Failed to create log for {log_name}: {e}")
