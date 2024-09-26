import os

from src import step, Pipeline
from src.models.model.huggingface.hugging_face_model_context import (
    HuggingFaceModelContext,
)
from src.models.steps.model_loading.common.CLIP.clip_model_context_loader import (
    clip_load_model,
)


@step
def clip_model_context_loader(
    model_context: HuggingFaceModelContext,
) -> HuggingFaceModelContext:
    context = Pipeline.get_active_context()
    if (
        model_context.pretrained_weights_path
        and os.path.exists(model_context.pretrained_weights_path)
        and model_context.config_path
        and os.path.exists(model_context.config_path)
    ):
        loaded_model, loaded_processor = clip_load_model(
            model_name=model_context.hugging_face_model_name,
            device=context.processing_parameters.device,
        )
        model_context.set_loaded_model(loaded_model)
        model_context.set_loaded_processor(loaded_processor)
    else:
        raise FileNotFoundError(
            f"Pretrained weights or config file not found at path: {model_context.pretrained_weights_path} or {model_context.config_path}"
        )
    return model_context
