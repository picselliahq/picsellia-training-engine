from src import step
from src.models.model.huggingface.hugging_face_model_context import (
    HuggingFaceModelContext,
)
from src.models.steps.model_loading.common.CLIP.clip_model_context_loader import (
    clip_load_model,
)


@step
def clip_model_context_loader(
    model_context: HuggingFaceModelContext,
    device: str = "cuda:0",
) -> HuggingFaceModelContext:
    loaded_model, loaded_processor = clip_load_model(
        model_name=model_context.hugging_face_model_name,
        device=device,
    )
    model_context.set_loaded_model(loaded_model)
    model_context.set_loaded_processor(loaded_processor)
    return model_context
