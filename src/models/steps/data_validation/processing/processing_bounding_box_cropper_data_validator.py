from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.object_detection_dataset_context_validator import (
    ObjectDetectionDatasetContextValidator,
)

from picsellia import Client


class ProcessingBoundingBoxCropperDataValidator(ObjectDetectionDatasetContextValidator):
    def __init__(
        self,
        dataset_context: DatasetContext,
        client: Client,
        label_name_to_extract: str,
        datalake: str,
    ):
        super().__init__(dataset_context=dataset_context)
        self.client = client
        self.label_name_to_extract = label_name_to_extract
        self.datalake = datalake

    def _validate_label_name_to_extract(self) -> None:
        """
        Validate that the label name to extract is present in the labelmap.

        Raises:
            ValueError: If the label name to extract is not present in the labelmap.
        """
        if self.label_name_to_extract not in self.dataset_context.labelmap:
            raise ValueError(
                f"Label name {self.label_name_to_extract} is not present in the labelmap"
            )

    def _validate_datalake(self) -> None:
        """
        Validate that the datalake is valid.

        Raises:
            ValueError: If the datalake is not valid.
        """
        datalakes_name = [datalake.name for datalake in self.client.list_datalakes()]
        if self.datalake not in datalakes_name:
            raise ValueError(
                f"Datalake {self.datalake} is not valid, available datalakes are {datalakes_name}"
            )

    def validate(self) -> None:
        super().validate()
        self._validate_label_name_to_extract()
        self._validate_datalake()
