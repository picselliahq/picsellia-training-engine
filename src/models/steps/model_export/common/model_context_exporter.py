import os
from abc import abstractmethod
from typing import Any

from picsellia import Experiment
from src.models.model.common.model_context import TModelContext


class ModelContextExporter:
    """
    Base class for exporting and saving a model context.

    This class serves as a base for exporting a model and saving it to an experiment.
    It provides an abstract method `export_model_context` for subclasses to implement
    specific export logic, and a concrete method `save_model_to_experiment` for saving
    the exported model to the experiment.

    Attributes:
        model_context (ModelContext): The context of the model to be exported.
        experiment (Experiment): The experiment to which the model is related.
    """

    def __init__(self, model_context: TModelContext, experiment: Experiment):
        """
        Initializes the ModelContextExporter with the given model context and experiment.

        Args:
            model_context (ModelContext): The model context containing the model's details.
            experiment (Experiment): The experiment object where the model will be exported.
        """
        self.model_context = model_context
        self.experiment = experiment

    @abstractmethod
    def export_model_context(
        self,
        exported_model_destination_path: str,
        export_format: str,
        hyperparameters: Any,
    ):
        """
        Abstract method to export the model context.

        This method should be implemented by subclasses to define the logic for exporting
        the model context in the specified format.

        Args:
            exported_model_destination_path (str): The destination path where the exported model will be saved.
            export_format (str): The format in which the model should be exported.
        """
        pass

    def save_model_to_experiment(
        self, exported_weights_dir: str, exported_weights_name: str
    ):
        """
        Saves the exported model to the experiment.

        This method takes the directory where the model was exported and uploads it to the
        associated experiment. If multiple files exist in the directory, they are zipped before uploading.

        Args:
            exported_weights_dir (str): The directory where the exported model weights are stored.
            exported_weights_name (str): The name under which the model will be stored in the experiment.

        Returns:
            ModelContext: The updated model context after saving the model.

        Raises:
            ValueError: If no model files are found in the exported model directory.
        """
        exported_files = os.listdir(exported_weights_dir)
        if not exported_files:
            raise ValueError("No model files found in the exported model directory")

        if len(exported_files) > 1:
            self.experiment.store(
                name=exported_weights_name,
                path=exported_weights_dir,
                do_zip=True,
            )
        else:
            self.experiment.store(
                name=exported_weights_name,
                path=os.path.join(
                    exported_weights_dir,
                    exported_files[0],
                ),
            )
