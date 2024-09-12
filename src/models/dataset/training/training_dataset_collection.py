import os
from typing import Generic, List

from src.models.dataset.common.dataset_context import TDatasetContext


class TrainingDatasetCollection(Generic[TDatasetContext]):
    """
    A collection of dataset contexts for different splits of a dataset.

    This class aggregates dataset contexts for the common splits used in machine learning projects:
    training, validation, and testing. It provides a convenient way to access and manipulate these
    dataset contexts as a unified object. The class supports direct access to individual dataset
    contexts, iteration over all contexts, and collective operations on all contexts, such as downloading
    assets.

    Attributes:
        datasets (dict): A dictionary containing dataset contexts, where keys are dataset names.
        dataset_path (str): The common file path for all dataset splits.
    """

    def __init__(self, datasets: List[TDatasetContext]):
        """
        Initializes the collection with a list of dataset contexts.

        Args:
            datasets (List[TDatasetContext]): A list of dataset contexts for different splits (train, val, test).
        """
        self.datasets = {dataset.dataset_name: dataset for dataset in datasets}
        self.dataset_path = self.get_common_path()

    def get_common_path(self) -> str:
        """
        Computes the common file path for all dataset splits.

        Returns:
            str: The common path shared by all dataset contexts.
        """
        return os.path.commonpath(
            [dataset_context.dataset_path for dataset_context in self.datasets.values()]
        )

    def __getitem__(self, key: str) -> TDatasetContext:
        """
        Retrieves a dataset context by its name.

        Args:
            key (str): The name of the dataset context.

        Returns:
            TDatasetContext: The dataset context corresponding to the given name.
        """
        return self.datasets[key]

    def __setitem__(self, key: str, value: TDatasetContext):
        """
        Sets or updates a dataset context in the collection.

        Args:
            key (str): The name of the dataset context to update or add.
            value (TDatasetContext): The dataset context object to associate with the given name.
        """
        self.datasets[key] = value

    def __iter__(self):
        """
        Iterates over all dataset contexts in the collection.

        Returns:
            Iterator: An iterator over the dataset contexts.
        """
        return iter(self.datasets.values())

    def download_all(self) -> None:
        """
        Downloads all assets and annotations for every dataset context in the collection.

        For each dataset context, it downloads the assets to the corresponding image directory and
        builds the COCO file from the annotations.
        """
        for dataset_context in self:
            dataset_context.download_assets(image_dir=dataset_context.image_dir)
            dataset_context.download_and_build_coco_file(
                annotations_dir=dataset_context.annotations_dir
            )
