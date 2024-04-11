from src import step
from src.models.dataset.dataset_collection import DatasetCollection
from src.steps.data_validation.utils.classification_dataset_validator import (
    ClassificationDatasetValidator,
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
    validator = ClassificationDatasetValidator(dataset_collection)
    validator.validate()
