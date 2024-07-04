from abc import abstractmethod

from src.models.model.model_context import ModelContext

from picsellia import Experiment


class ModelContextExporter:
    def __init__(self, model_context: ModelContext, experiment: Experiment):
        self.model_context = model_context
        self.experiment = experiment

    @abstractmethod
    def export_model(self):
        pass

    def save_model_to_experiment(self):
        saved_model_name = (
            f"{self.model_context.prefix_model_name}-model-latest"
            if self.model_context.prefix_model_name
            else "model-latest"
        )
        self.experiment.store(
            name=saved_model_name, path=self.model_context.inference_model_path
        )
        return self.model_context

    def export_and_save_model(self):
        self.export_model()
        self.save_model_to_experiment()
