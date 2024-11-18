from picsellia import DatasetVersion

from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.dataset_context import DatasetContext


class ProcessingDatasetCollectionExtractor:
    """
    A class responsible for extracting and managing input and output datasets as a collection.

    This class handles the creation of a `DatasetCollection` from two dataset versions: one for input data and one
    for output data. It organizes the datasets into contexts and returns them in a unified collection.

    Attributes:
        input_dataset_version (DatasetVersion): The input dataset version from Picsellia.
        output_dataset_version (DatasetVersion): The output dataset version from Picsellia.
    """

    def __init__(
        self,
        input_dataset_version: DatasetVersion,
        output_dataset_version: DatasetVersion,
    ):
        """
        Initializes the ProcessingDatasetCollectionExtractor with input and output dataset versions.

        Args:
            input_dataset_version (DatasetVersion): The version of the input dataset to be processed.
            output_dataset_version (DatasetVersion): The version of the output dataset to be processed.
        """
        self.input_dataset_version = input_dataset_version
        self.output_dataset_version = output_dataset_version

    def get_dataset_collection(self) -> DatasetCollection:
        """
        Creates and returns a DatasetCollection with input and output dataset contexts.

        This method organizes the input and output datasets into `DatasetContext` objects and returns
        them as part of a `DatasetCollection`. Each dataset context contains its assets and metadata.

        Returns:
            DatasetCollection: A collection of input and output datasets as DatasetContext objects.
        """
        input_dataset_context = DatasetContext(
            dataset_name="input",
            dataset_version=self.input_dataset_version,
            assets=self.input_dataset_version.list_assets(),
            labelmap=None,
        )
        output_dataset_context = DatasetContext(
            dataset_name="output",
            dataset_version=self.output_dataset_version,
            assets=self.output_dataset_version.list_assets(),
            labelmap=None,
        )
        return DatasetCollection([input_dataset_context, output_dataset_context])
