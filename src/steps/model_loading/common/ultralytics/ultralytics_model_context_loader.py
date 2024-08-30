import os.path

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_loading.ultralytics.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def ultralytics_model_context_loader(model_context: ModelContext) -> ModelContext:
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters
    ] = Pipeline.get_active_context()
    if (
        model_context.pretrained_model_path
        and os.path.exists(model_context.pretrained_model_path)
        and isinstance(context.hyperparameters, UltralyticsHyperParameters)
    ):
        loaded_model = ultralytics_load_model(
            weights_path_to_load=model_context.pretrained_model_path,
            device=context.hyperparameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_context.pretrained_model_path}. Cannot load model."
        )
    return model_context
