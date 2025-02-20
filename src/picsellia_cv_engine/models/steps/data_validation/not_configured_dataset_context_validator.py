from src.picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from src.picsellia_cv_engine.models.steps.data_validation.dataset_context_validator import (
    DatasetContextValidator,
)


class NotConfiguredDatasetContextValidator(
    DatasetContextValidator[TBaseDatasetContext]
):
    def validate(self):
        """
        Validate the dataset context.

        Raises:
            ValueError: If the dataset context is not valid.
        """
        super().validate()
