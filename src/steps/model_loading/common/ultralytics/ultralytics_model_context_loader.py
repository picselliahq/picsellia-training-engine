from src import step
from src.models.model.common.model_context import ModelContext
from src.models.steps.model_loading.ultralytics.ultralytics_model_context_loader import (
    UltralyticsModelContextLoader,
)


@step
def ultralytics_model_context_loader(model_context: ModelContext) -> ModelContext:
    model_context_loader = UltralyticsModelContextLoader(model_context=model_context)
    model_context = model_context_loader.load_model()
    return model_context
