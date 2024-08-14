import os
from abc import abstractmethod

from src.models.model.model_context import ModelContext

from picsellia import Experiment


class ModelContextExporter:
    def __init__(self, model_context: ModelContext, experiment: Experiment):
        self.model_context = model_context
        self.experiment = experiment

    @abstractmethod
    def export_model_context(self):
        pass

    def save_model_to_experiment(self):
        saved_model_name = (
            f"{self.model_context.prefix_model_name}-model-latest"
            if self.model_context.prefix_model_name
            else "model-latest"
        )
        inference_model_files = os.listdir(self.model_context.inference_model_path)
        if not inference_model_files:
            raise ValueError("No model files found in inference model path")
        elif len(inference_model_files) > 1:
            self.experiment.store(
                name=saved_model_name,
                path=self.model_context.inference_model_path,
                do_zip=True,
            )
        else:
            self.experiment.store(
                name=saved_model_name,
                path=os.path.join(
                    self.model_context.inference_model_path, inference_model_files[0]
                ),
            )
        return self.model_context

    def export_and_save_model_context(self):
        self.export_model_context()
        self.model_context = self.save_model_to_experiment()
        return self.model_context
