import os
from typing import Optional
from picsellia import Experiment
from src.models.model.common.model_context import ModelContext


class TrainingModelContextExtractor:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.destination_path = os.path.join(os.getcwd(), self.experiment.name)

    def get_model_context(
        self,
        prefix_model_name: Optional[str] = None,
        pretrained_model_filename: str = "model-latest",
        config_filename: str = "config",
        destination_path: Optional[str] = None,
    ) -> ModelContext:
        """
        Retrieves the model context from the active Picsellia context.

        Args:
            prefix_model_name (Optional[str]): Prefix for model file names.
            pretrained_model_filename (str): Name of the pretrained model file on Picsellia.
            config_filename (str): Name of the config file on Picsellia.
            destination_path: The destination path to save the model files.

        Returns:
            - ModelContext: The model context object.
        """
        model_version = self.experiment.get_base_model_version()
        model_name = model_version.name
        if not destination_path:
            destination_path = os.path.join(os.getcwd(), self.experiment.name)
        return ModelContext(
            model_name=model_name,
            model_version=model_version,
            destination_path=destination_path,
            pretrained_model_filename=pretrained_model_filename,
            config_filename=config_filename,
            prefix_model_name=prefix_model_name,
        )
