from picsellia import Experiment

from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.model.common.model_context import ModelContext
from src.models.model.ultralytics.ultralytics_callbacks import UltralyticsCallbacks

from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)


class UltralyticsModelContextTrainer:
    def __init__(
        self,
        model_context: ModelContext,
        experiment: Experiment,
    ):
        self.model_context = model_context
        self.experiment = experiment
        self.callbacks = UltralyticsCallbacks(experiment=experiment)

    def train_model_context(
        self,
        dataset_collection: TrainingDatasetCollection,
        hyperparameters: UltralyticsHyperParameters,
    ):
        """
        Trains both the bounding box detection and text recognition models in the model collection.
        """
        callbacks = self.callbacks.get_callbacks()
        for callback_name, callback_function in callbacks.items():
            self.model_context.loaded_model.add_callback(
                event=callback_name, func=callback_function
            )
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
