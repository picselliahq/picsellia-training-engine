import os

from src import step, Pipeline
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_loading.common.MiniGPT.minigpt_model_context_loader import (
    minigpt_load_model,
)


@step
def minigpt_model_context_loader(model_context: ModelContext) -> ModelContext:
    context = Pipeline.get_active_context()
    if (
        model_context.pretrained_weights_path
        and os.path.exists(model_context.pretrained_weights_path)
        and model_context.config_path
        and os.path.exists(model_context.config_path)
    ):
        loaded_model = minigpt_load_model(
            weights_path_to_load=model_context.pretrained_weights_path,
            config_path_to_load=model_context.config_path,
            device=context.processing_parameters.device,
        )
        model_context.set_loaded_model(loaded_model)
    else:
        raise FileNotFoundError(
            f"Pretrained weights or config file not found at path: {model_context.pretrained_weights_path} or {model_context.config_path}"
        )
    return model_context