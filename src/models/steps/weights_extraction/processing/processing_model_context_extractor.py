import os
from typing import Optional

from picsellia import ModelVersion

from src.models.model.model_context import ModelContext


class ProcessingModelContextExtractor:
    def __init__(self, model_version: ModelVersion):
        self.model_version = model_version
        self.destination_path = os.path.join(os.getcwd(), self.model_version.name)

    def get_model_context(
            self, prefix_model_name: Optional[str] = None
    ) -> ModelContext:
        """
        Retrieves the model context from the active Picsellia context.

        Returns:
            - ModelContext: The model context object.
        """
        return ModelContext(
            model_name=self.model_version.name,
            model_version=self.model_version,
            destination_path=self.destination_path,
            prefix_model_name=prefix_model_name,
        )
