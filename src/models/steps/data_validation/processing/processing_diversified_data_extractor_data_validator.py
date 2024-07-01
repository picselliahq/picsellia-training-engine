from src.models.dataset.common.dataset_context import DatasetContext
from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class ProcessingDiversifiedDataExtractorDataValidator(DatasetContextValidator):
    def __init__(
        self,
        dataset_context: DatasetContext,
    ):
        super().__init__(dataset_context=dataset_context)

    def _validate_dataset_version_size(self) -> None:
        """
        Validate that the dataset version size is greater than .

        Raises:
            ValueError: If the dataset version size is less than or equal to 0.
        """
        dataset_version_size = self.dataset_context.dataset_version.sync()["size"]
        if dataset_version_size == 0:
            raise ValueError(
                "This dataset version cannot be diversified because it is empty. "
                "Please add some assets to the dataset version before running this processing."
            )

        elif dataset_version_size == 1:
            raise ValueError(
                "This dataset version has only one asset, therefore it cannot be diversified. "
                "Please add more assets to the dataset version before running this processing."
            )

    def validate(self) -> None:
        super().validate()
        self._validate_dataset_version_size()
