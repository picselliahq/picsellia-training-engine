import os
from typing import Optional

from picsellia import Experiment

from src.models.model.common.model_context import ModelContext


class TrainingModelContextExtractor:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.destination_path = os.path.join(os.getcwd(), self.experiment.name)

    def get_model_context(
        self, prefix_model_name: Optional[str] = None
    ) -> ModelContext:
        """
        Retrieves the model context from the active Picsellia context.

        Returns:
            - ModelContext: The model context object.
        """
        model_version = self.experiment.get_base_model_version()
        model_name = model_version.name
        return ModelContext(
            model_name=model_name,
            model_version=model_version,
            destination_path=self.destination_path,
            prefix_model_name=prefix_model_name,
        )
