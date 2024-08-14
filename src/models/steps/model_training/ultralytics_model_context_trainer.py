from picsellia import Experiment

from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.model.model_context import ModelContext

from src.models.parameters.common.hyper_parameters import UltralyticsHyperParameters


class UltralyticsModelContextTrainer:
    def __init__(
        self,
        model_context: ModelContext,
        experiment: Experiment,
    ):
        self.model_context = model_context
        self.experiment = experiment

    def train_model_context(
        self,
        dataset_collection: TrainingDatasetCollection,
        hyperparameters: UltralyticsHyperParameters,
    ):
        """
        Trains both the bounding box detection and text recognition models in the model collection.
        """
        if hyperparameters.epochs > 0:
            self.model_context.loaded_model.train(
                data=dataset_collection.dataset_path,
                epochs=hyperparameters.epochs,
                batch=hyperparameters.batch_size,
                imgsz=hyperparameters.image_size,
                device=hyperparameters.device,
                project=self.model_context.results_path,
                name=self.model_context.model_name,
            )
        return self.model_context
