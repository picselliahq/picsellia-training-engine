from typing import Type

from src.models.dataset.common.dataset_collection import DatasetCollection
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
        dataset_context_validator (Type[DatasetContextValidator]): The validator class for individual dataset contexts.
    """

    def __init__(
        self,
        dataset_collection: DatasetCollection,
        dataset_context_validator: Type[DatasetContextValidator],
    ):
        """
        Initializes the DatasetCollectionValidator with a dataset collection to validate.

        Parameters:
            dataset_collection (DatasetCollection): The dataset collection to validate.
            dataset_context_validator (Type[DatasetContextValidator]): The class used to validate individual dataset contexts.
        """
        self.dataset_collection = dataset_collection
        self.dataset_context_validator = dataset_context_validator

    def validate(self, fix_annotation: bool = False) -> None:
        """
        Validates the dataset collection.

        Iterates through the dataset contexts in the collection and applies the context validator
        for each dataset.
        """
        for dataset_context in self.dataset_collection:
            validator = self.dataset_context_validator(
                dataset_context=dataset_context, fix_annotation=fix_annotation
            )
            validator.validate()
