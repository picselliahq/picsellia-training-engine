from src import step
from src.steps.data_extraction.utils.dataset_collection import DatasetCollection
from src.steps.data_validation.utils.dataset_validator import (
    ClassificationDatasetValidator,
)


@step
def classification_data_validator(dataset_collection: DatasetCollection):
    validator = ClassificationDatasetValidator(dataset_collection)
    validator.validate()
