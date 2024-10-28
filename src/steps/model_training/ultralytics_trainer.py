from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.common.dataset_collection import DatasetCollection
from src.models.model.common.model_context import ModelContext
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
    model_context: ModelContext, dataset_collection: DatasetCollection
) -> ModelContext:
    """
    Trains an Ultralytics model on the provided dataset collection.

    This function retrieves the active training context and initializes an `UltralyticsModelContextTrainer`.
    It trains the Ultralytics model using the provided dataset collection, applying the hyperparameters and
    augmentation parameters specified in the context. After training, the updated model context is returned.

    Args:
        model_context (ModelContext): The context containing the Ultralytics model to be trained.
        dataset_collection (DatasetCollection): The dataset collection to be used for training the model.

    Returns:
        ModelContext: The updated model context after training.
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

    return model_context
