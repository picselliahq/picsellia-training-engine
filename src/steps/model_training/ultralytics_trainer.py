import os.path

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_training.common.ultralytics_model_context_trainer import (
    UltralyticsModelContextTrainer,
)


@step
def ultralytics_model_context_trainer(
    model_context: UltralyticsModelContext, dataset_collection: DatasetCollection
) -> UltralyticsModelContext:
    """
    Trains an Ultralytics model on the provided dataset collection.

    This function retrieves the active training context and initializes an `UltralyticsModelContextTrainer`.
    It trains the Ultralytics model using the provided dataset collection, applying the hyperparameters and
    augmentation parameters specified in the context. After training, the updated model context is returned.

    Args:
        model_context (UltralyticsModelContext): The context containing the Ultralytics model to be trained.
        dataset_collection (DatasetCollection): The dataset collection to be used for training the model.

    Returns:
        UltralyticsModelContext: The updated model context after training.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    model_context_trainer = UltralyticsModelContextTrainer(
        model_context=model_context,
        experiment=context.experiment,
    )

    model_context = model_context_trainer.train_model_context(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        augmentation_parameters=context.augmentation_parameters,
    )

    model_context.set_latest_run_dir()
    model_context.set_trained_weights_path()
    if not model_context.trained_weights_path or not os.path.exists(
        model_context.trained_weights_path
    ):
        raise FileNotFoundError(
            f"Trained weights not found at {model_context.trained_weights_path}"
        )
    model_context.save_artifact_to_experiment(
        experiment=context.experiment,
        artifact_name="best-model",
        artifact_path=model_context.trained_weights_path,
    )

    return model_context
