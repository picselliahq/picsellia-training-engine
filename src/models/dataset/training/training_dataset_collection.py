import os
from typing import Generic, List

from src.models.dataset.common.dataset_context import TDatasetContext

from picsellia import Experiment


class TrainingDatasetCollection(Generic[TDatasetContext]):
    """
    A collection of dataset contexts for different splits of a dataset.

    This class aggregates dataset contexts for the common splits used in machine learning projects:
    training, validation, and testing. It provides a convenient way to access and manipulate these
    dataset contexts as a unified object. The class supports direct access to individual dataset
    contexts, iteration over all contexts, and collective operations on all contexts, such as downloading
    assets.

    Attributes:
        train (DatasetContext): The dataset context for the training split.
        val (DatasetContext): The dataset context for the validation split.
        test (DatasetContext): The dataset context for the testing split.
    """

    def __init__(
        self,
        datasets: List[TDatasetContext],
    ):
        self.datasets = {dataset.dataset_name: dataset for dataset in datasets}
        self.dataset_path = self._unify_dataset_paths(
            os.path.join(self.datasets["train"].destination_path, "dataset")
        )

    def _unify_dataset_paths(self, dataset_path: str) -> str:
        for dataset in self:
            dataset.update_destination_path(dataset_path)
        return dataset_path

    def __getitem__(self, key: str) -> TDatasetContext:
        return self.datasets[key]

    def __setitem__(self, key: str, value: TDatasetContext):
        self.datasets[key] = value

    def __iter__(self):
        return iter(self.datasets.values())

    def download_all(self):
        for dataset_context in self:
            dataset_context.download_assets()
            dataset_context.download_and_build_coco_file()

    def log_labelmap(self, experiment: Experiment):
        labelmap_to_log = {
            str(i): label
            for i, label in enumerate(self.datasets["train"].labelmap.keys())
        }
        experiment.log("labelmap", labelmap_to_log, "labelmap", replace=True)
