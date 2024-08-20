import os
from typing import Optional

from picsellia import DatasetVersion

from src.models.dataset.common.dataset_context import DatasetContext


class ProcessingDatasetContextExtractor:
    def __init__(
        self,
        dataset_version: DatasetVersion,
        job_id: Optional[str] = None,
        use_id: Optional[bool] = True,
    ):
        """
        Initializes a DatasetHandler with the input dataset version and destination path.

        Args:
            dataset_version (DatasetVersion): The dataset version to be processed.
        """
        self.dataset_version = dataset_version
        if not job_id:
            self.destination_path = os.path.join(os.getcwd(), "current_job")
        else:
            self.destination_path = os.path.join(os.getcwd(), str(job_id))
        self.use_id = use_id

    def get_dataset_context(self, skip_asset_listing: bool = False) -> DatasetContext:
        """
        Retrieves the input dataset version and prepares a dataset context for extraction.

        This method downloads all necessary assets and annotations from the dataset version
        and organizes them into a dataset context for extraction.

        Args:
            skip_asset_listing (bool): Whether to skip listing the dataset's assets.

        Returns:
            A dataset context prepared for extraction, including all assets and annotations downloaded.
        """
        return DatasetContext(
            dataset_name="dataset_to_process",
            dataset_version=self.dataset_version,
            destination_path=self.destination_path,
            multi_asset=None,
            labelmap=None,
            skip_asset_listing=skip_asset_listing,
            use_id=self.use_id,
        )
