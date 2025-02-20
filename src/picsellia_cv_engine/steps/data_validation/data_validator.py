from src.picsellia_cv_engine import step
from src.picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from src.picsellia_cv_engine.models.dataset.dataset_collection import (
    DatasetCollection,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_collection_validator import (
    DatasetCollectionValidator,
    DatasetContextValidator,
)


@step
def validate_training_data(dataset_collection: DatasetCollection):
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
def validate_processing_data(dataset_context: TBaseDatasetContext):
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
