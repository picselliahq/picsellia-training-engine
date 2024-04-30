from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.pipelines.bounding_box_cropper.utils.bounding_box_cropper_parameters import (
    BoundingBoxCropperParameters,
)
from src.steps.data_validation.utils.object_detection_dataset_context_validator import (
    ObjectDetectionDatasetContextValidator,
)

from picsellia import Client


class BoundingBoxCropperDataValidator(ObjectDetectionDatasetContextValidator):
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


@step
def bounding_box_cropper_data_validator(
    dataset_context: DatasetContext,
) -> None:
    context: PicselliaProcessingContext[
        BoundingBoxCropperParameters
    ] = Pipeline.get_active_context()

    validator = BoundingBoxCropperDataValidator(
        dataset_context=dataset_context,
        client=context.client,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
        datalake=context.processing_parameters.datalake,
    )
    validator.validate()
