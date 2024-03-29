from src import step
from src.models.dataset.dataset_collection import DatasetCollection
from src.steps.data_validation.utils.dataset_validator import DatasetValidator


@step
def data_validator(dataset_collection: DatasetCollection):
    """
    Performs common validations on a dataset collection.

    Initializes a DatasetValidator with the provided dataset collection and invokes its `validate` method
    to carry out a series of general validations. These include verifying that all images have been correctly extracted,
    checking for image corruption, ensuring images are in the correct format, and more. These validations are
    applicable across different types of machine learning tasks and datasets.

    Args:
        dataset_collection (DatasetCollection): The collection of datasets to be validated, encompassing
                                                potentially multiple dataset splits (e.g., training, validation, testing).

    """
    validator = DatasetValidator(dataset_collection)
    validator.validate()
