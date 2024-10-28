from typing import Optional

from picsellia import DatasetVersion

from src.models.dataset.common.dataset_context import DatasetContext
from tf2.experiment.main import assets


class ProcessingDatasetContextExtractor:
    """
    A class responsible for extracting and managing a dataset context for a specific dataset version.

    This class takes a Picsellia `DatasetVersion` and prepares a corresponding `DatasetContext`,
    which includes assets and annotations that are ready for processing or further extraction.

    Attributes:
        dataset_version (DatasetVersion): The dataset version from Picsellia that will be processed.
        use_id (Optional[bool]): Whether to use asset IDs when organizing the dataset's assets.
    """

    def __init__(
        self,
        dataset_version: DatasetVersion,
        use_id: Optional[bool] = True,
    ):
        """
        Initializes the ProcessingDatasetContextExtractor with a dataset version and an optional use_id flag.

        Args:
            dataset_version (DatasetVersion): The version of the dataset to be processed and extracted.
            use_id (Optional[bool]): If True, uses asset IDs for organizing file paths. Defaults to True.
        """
        self.dataset_version = dataset_version
        self.use_id = use_id

    def get_dataset_context(self) -> DatasetContext:
        """
        Retrieves the dataset context by downloading assets and annotations from the specified dataset version.

        This method prepares a `DatasetContext` for further processing, ensuring that the assets from
        the dataset version are available.

        Returns:
            DatasetContext: A dataset context that contains all assets and metadata required for extraction.
        """
        return DatasetContext(
            dataset_name="input",
            dataset_version=self.dataset_version,
            assets=assets,
            labelmap=None,
        )
