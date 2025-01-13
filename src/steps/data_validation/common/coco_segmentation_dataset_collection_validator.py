from src import step
from src.models.dataset.common.coco_dataset_context import CocoDatasetContext
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.models.steps.data_validation.common.coco_segmentation_dataset_context_validator import (
    CocoSegmentationDatasetContextValidator,
)


@step
def coco_segmentation_dataset_collection_validator(
    dataset_collection: DatasetCollection[CocoDatasetContext],
    fix_annotation: bool = False,
) -> None:
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=CocoSegmentationDatasetContextValidator,
    )
    validator.validate(fix_annotation=fix_annotation)
