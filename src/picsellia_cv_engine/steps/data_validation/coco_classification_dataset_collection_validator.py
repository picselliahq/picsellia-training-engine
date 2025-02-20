from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.dataset.coco_dataset_context import (
    CocoDatasetContext,
)
from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from src.picsellia_cv_engine.models.steps.data_validation.coco_classification_dataset_context_validator import (
    CocoClassificationDatasetContextValidator,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_collection_validator import (
    DatasetCollectionValidator,
)


@step
def validate_coco_classification_dataset_collection(
    dataset_collection: DatasetCollection[CocoDatasetContext],
) -> None:
    """
    Validates a dataset collection for classification tasks.

    This function initializes a ClassificationDatasetValidator with the provided dataset collection,
    then calls its `validate` method to perform validations specific to classification datasets. This includes
    checks for proper label mapping, sufficient images per class, and other classification-specific requirements.

    Args:
        dataset_collection (DatasetCollection): The collection of datasets to be validated, typically including
                                                training, validation, and testing splits.
    """
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=CocoClassificationDatasetContextValidator,
    )
    validator.validate()
