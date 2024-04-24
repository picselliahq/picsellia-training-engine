from src import step
from src.models.dataset.dataset_collection import DatasetCollection
from src.steps.data_validation.utils.classification_dataset_context_validator import (
    ClassificationDatasetContextValidator,
)
from src.steps.data_validation.utils.dataset_collection_validator import (
    DatasetCollectionValidator,
)


@step
def classification_data_validator(dataset_collection: DatasetCollection):
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
        dataset_context_validator=ClassificationDatasetContextValidator,
    )
    validator.validate()
