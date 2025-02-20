from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from src.picsellia_cv_engine.models.dataset.yolo_dataset_context import (
    YoloDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.picsellia_cv_engine.models.steps.data_validation.yolo_segmentation_dataset_context_validator import (
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
