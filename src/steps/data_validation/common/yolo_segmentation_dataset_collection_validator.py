from src import step
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.dataset.common.yolo_dataset_context import YoloDatasetContext
from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.models.steps.data_validation.common.yolo_segmentation_dataset_context_validator import (
    YoloSegmentationDatasetContextValidator,
)


@step
def yolo_segmentation_dataset_collection_validator(
    dataset_collection: DatasetCollection[YoloDatasetContext],
    fix_annotation: bool = False,
) -> None:
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=YoloSegmentationDatasetContextValidator,
    )
    validator.validate(fix_annotation=fix_annotation)
