import os
from abc import abstractmethod

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

    def save_model_to_experiment(
        self, exported_weights_dir: str, exported_weights_name: str
    ):
        """
        Saves the exported model to the experiment.

        Returns:
            ModelContext: The updated model context after saving the model.

        Raises:
            ValueError: If no model files are found in the inference model path.
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
