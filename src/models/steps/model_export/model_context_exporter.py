import os
from abc import abstractmethod
from typing import List

from picsellia import Experiment
from src.models.model.common.model_context import ModelContext


class ModelContextExporter:
    """
    Base class for exporting and saving a model context.

    Attributes:
        model_context (ModelContext): The context of the model to be exported.
        experiment (Experiment): The experiment to which the model is related.
    """

    def __init__(self, model_context: ModelContext, experiment: Experiment):
        self.model_context = model_context
        self.experiment = experiment

    @abstractmethod
    def export_model_context(
        self, exported_model_destination_path: str, export_format: str
    ):
        """
        Abstract method to be implemented by subclasses to define how the model context is exported.
        """
        pass

    def save_model_to_experiment(self) -> ModelContext:
        """
        Saves the exported model to the experiment.

        Returns:
            ModelContext: The updated model context after saving the model.

        Raises:
            ValueError: If no model files are found in the inference model path.
        """
        saved_model_name = (
            f"{self.model_context.prefix_model_name}-model-latest"
            if self.model_context.prefix_model_name
            else "model-latest"
        )

        inference_model_files = os.listdir(self.model_context.inference_model_dir)
        if not inference_model_files:
            raise ValueError("No model files found in inference model path")

        self._store_model_files(saved_model_name, inference_model_files)
        return self.model_context

    def _store_model_files(
        self, saved_model_name: str, inference_model_files: List[str]
    ) -> None:
        """
        Stores the model files in the experiment.

        Args:
            saved_model_name (str): The name under which the model will be saved.
            inference_model_files (list): The list of files in the inference model path.
        """
        if len(inference_model_files) > 1:
            self.experiment.store(
                name=saved_model_name,
                path=self.model_context.inference_model_dir,
                do_zip=True,
            )
        else:
            self.experiment.store(
                name=saved_model_name,
                path=os.path.join(
                    str(self.model_context.inference_model_dir),
                    inference_model_files[0],
                ),
            )
