from src import Pipeline
from src import step
from src.models.contexts.picsellia_context import PicselliaProcessingContext
from src.models.dataset.dataset_context import DatasetContext
from src.models.parameters.processing.bounding_box_cropper_parameters import (
    BoundingBoxCropperParameters,
)

from src.models.steps.data_validation.processing.bounding_box_cropper_data_validator import (
    BoundingBoxCropperDataValidator,
)


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
