# type: ignore
from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.dataset.training.training_dataset_collection import (
    TrainingDatasetCollection,
)
from src.models.model.common.model_context import ModelContext
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_training.ultralytics_model_context_trainer import (
    UltralyticsModelContextTrainer,
)


@step
def ultralytics_trainer(
    model_context: ModelContext, dataset_collection: TrainingDatasetCollection
) -> ModelContext:
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters
    ] = Pipeline.get_active_context()
    model_trainer = UltralyticsModelContextTrainer(
        model_context=model_context,
        experiment=context.experiment,
    )
    model_context = model_trainer.train_model_context(
        dataset_collection=dataset_collection,
        hyperparameters=context.hyperparameters,
        augmentation_parameters=context.augmentation_parameters,
    )
    return model_context
