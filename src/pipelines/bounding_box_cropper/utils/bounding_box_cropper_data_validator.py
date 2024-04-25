from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.pipelines.bounding_box_cropper.utils.bounding_box_cropper_parameters import (
    BoundingBoxCropperParameters,
)
from src.steps.data_validation.utils.dataset_collection_validator import (
    DatasetContextValidator,
)


class BoundingBoxCropperDataValidator(DatasetContextValidator):
    def __init__(self, dataset_context: DatasetContext, label_name_to_extract: str):
        super().__init__(dataset_context=dataset_context)
        self.label_name_to_extract = label_name_to_extract

    def _validate_label_name_to_extract(self) -> None:
        # Validate that the label name to extract is present in the labelmap
        labelmap = self.dataset_context.labelmap
        if not self.dataset_context.labelmap:
            raise ValueError("Labelmap is missing from the dataset context")
        if self.label_name_to_extract not in labelmap:
            raise ValueError(
                f"Label name {self.label_name_to_extract} is not present in the labelmap"
            )

    def validate(self) -> None:
        super().validate()
        self._validate_label_name_to_extract()


@step
def bounding_box_cropper_data_validator(
    dataset_context: DatasetContext,
) -> None:
    context: PicselliaProcessingContext[
        BoundingBoxCropperParameters
    ] = Pipeline.get_active_context()

    validator = BoundingBoxCropperDataValidator(
        dataset_context=dataset_context,
        label_name_to_extract=context.processing_parameters.label_name_to_extract,
    )
    validator.validate()
