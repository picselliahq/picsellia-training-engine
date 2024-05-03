from picsellia import DatasetVersion, Client

from src.models.dataset.dataset_context import DatasetContext


class DiversifiedDataExtractorProcessing:
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(
        self,
        client: Client,
        input_dataset_context: DatasetContext,
        output_dataset_version: DatasetVersion,
    ):
        self.client = client
        self.input_dataset_context = input_dataset_context
        self.output_dataset_context = DatasetContext(
            dataset_name="processed_dataset",
            dataset_version=output_dataset_version,
            destination_path="",
            multi_asset=None,
            labelmap=None,
        )

    def process(self) -> None:
        pass
