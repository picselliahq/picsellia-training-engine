from src.models.steps.data_validation.common.dataset_context_validator import (
    DatasetContextValidator,
)


class NotConfiguredDatasetContextValidator(DatasetContextValidator):
    def validate(self):
        """
        Validate the dataset context.

        Raises:
            ValueError: If the dataset context is not valid.
        """
        super().validate()
