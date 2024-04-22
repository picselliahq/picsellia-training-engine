import os

from picsellia import DatasetVersion

from src.models.dataset.dataset_context import DatasetContext


class ProcessingDatasetHandler:
    def __init__(self, job_id: str, dataset_version: DatasetVersion):
        """
        Initializes a DatasetHandler with the input dataset version and destination path.

        Args:
            dataset_version (DatasetVersion): The dataset version to be processed.
        """
        self.dataset_version = dataset_version
        self.destination_path = os.path.join(os.getcwd(), str(job_id))

    def get_dataset_context(self) -> DatasetContext:
        """
        Retrieves the input dataset version and prepares a dataset context for extraction.

        This method downloads all necessary assets and annotations from the dataset version
        and organizes them into a dataset context for extraction.

        Returns:
            - DatasetContext: A dataset context prepared for extraction, including all assets and annotations downloaded.
        """
        return DatasetContext(
            dataset_name="dataset_to_process",
            dataset_version=self.dataset_version,
            destination_path=self.destination_path,
            multi_asset=None,
            labelmap=None,
        )
