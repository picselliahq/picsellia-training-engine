from src import step
from src.models.dataset.common.dataset_context import DatasetContext
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.steps.data_validation.common.dataset_collection_validator import (
    DatasetCollectionValidator,
    DatasetContextValidator,
)


@step
def training_data_validator(dataset_collection: TrainingDatasetCollection):
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
    validator = DatasetCollectionValidator(
        dataset_collection=dataset_collection,
        dataset_context_validator=DatasetContextValidator,
    )
    validator.validate()


@step
def processing_data_validator(dataset_context: DatasetContext):
    """
    Performs common validations on a dataset context.

    Initializes a DatasetContextValidator with the provided dataset context and invokes its `validate` method
    to carry out a series of general validations. These include verifying that all images have been correctly extracted,
    checking for image corruption, ensuring images are in the correct format, and more. These validations are
    applicable across different types of machine learning tasks and datasets.

    Args:
        dataset_context (DatasetContext): The dataset context to be validated.

    """
    validator = DatasetContextValidator(dataset_context=dataset_context)
    validator.validate()
