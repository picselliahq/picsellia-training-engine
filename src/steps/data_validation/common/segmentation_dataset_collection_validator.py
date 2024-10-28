from src import step
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetCollectionValidator,
)
from src.models.steps.data_validation.common.segmentation_dataset_context_validator import (
    SegmentationDatasetContextValidator,
)


@step
def segmentation_dataset_collection_validator(
    dataset_collection: DatasetCollection, fix_annotation: bool = False
) -> None:
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=SegmentationDatasetContextValidator,
    )
    validator.validate(fix_annotation=fix_annotation)
