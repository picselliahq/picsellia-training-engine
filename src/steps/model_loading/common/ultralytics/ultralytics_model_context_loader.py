import os.path

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.parameters.common.export_parameters import ExportParameters
from src.models.parameters.training.ultralytics.ultralytics_augmentation_parameters import (
    UltralyticsAugmentationParameters,
)
from src.models.parameters.training.ultralytics.ultralytics_hyper_parameters import (
    UltralyticsHyperParameters,
)
from src.models.steps.model_loading.common.ultralytics.ultralytics_model_context_loader import (
    ultralytics_load_model,
)


@step
def ultralytics_model_context_loader(
    model_context: ModelContext, weights_path_to_load: str
) -> ModelContext:
    """
    Loads an Ultralytics model from pretrained weights if available.

    This function retrieves the active training context and attempts to load the Ultralytics model from
    the pretrained weights specified in the model context. If the pretrained weights file exists, the model
    is loaded onto the specified device. If the pretrained weights are not found, a `FileNotFoundError` is raised.

    Args:
        model_context (ModelContext): The model context containing the path to the pretrained weights and
                                      other model-related configurations.

    Returns:
        ModelContext: The updated model context with the loaded model.

    Raises:
        FileNotFoundError: If the pretrained weights file is not found at the specified path in the model context.
    """
    context: PicselliaTrainingContext[
        UltralyticsHyperParameters, UltralyticsAugmentationParameters, ExportParameters
    ] = Pipeline.get_active_context()

    if os.path.exists(weights_path_to_load):
        loaded_model = ultralytics_load_model(
            weights_path_to_load=weights_path_to_load,
            device=context.hyperparameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {weights_path_to_load}. Cannot load model."
        )

    return model_context
