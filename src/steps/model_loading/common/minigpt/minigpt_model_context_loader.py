import os.path

from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_loading.common.MiniGPT.minigpt_model_context_loader import minigpt_load_model


@step
def minigpt_model_context_loader(model_context: ModelContext) -> ModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    if model_context.pretrained_weights_path and os.path.exists(
        model_context.pretrained_weights_path
    ):
        loaded_model = minigpt_load_model(
            weights_path_to_load=model_context.pretrained_weights_path,
            device=context.hyperparameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained model file not found at {model_context.pretrained_weights_path}. Cannot load model."
        )
    return model_context
