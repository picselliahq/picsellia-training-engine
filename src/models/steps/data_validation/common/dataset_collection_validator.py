from typing import Type

from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class DatasetCollectionValidator:
    """
    Validates various aspects of a dataset collection.

    This class performs common validation tasks for dataset collections, including checking for image extraction
    completeness, image format, image corruption, and annotation integrity.

    Attributes:
        dataset_collection (DatasetCollection): The dataset collection to validate.
    """

    def __init__(
        self,
        dataset_collection: TrainingDatasetCollection,
        dataset_context_validator: Type[DatasetContextValidator],
    ):
        """
        Initializes the DatasetCollectionValidator with a dataset collection to validate.

        Parameters:
            dataset_collection (DatasetCollection): The dataset collection to validate.
        """
        self.dataset_collection = dataset_collection
        self.dataset_context_validator = dataset_context_validator

    def validate(self) -> None:
        """
        Validates the dataset collection.
        """
        for dataset_context in self.dataset_collection:
            validator = self.dataset_context_validator(dataset_context)
            validator.validate()
