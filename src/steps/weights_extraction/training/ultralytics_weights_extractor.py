from src import step, Pipeline
from src.models.contexts.training.picsellia_training_context import (
    PicselliaTrainingContext,
)
from src.models.model.ultralytics.ultralytics_model_context import (
    UltralyticsModelContext,
)
from src.models.steps.weights_extraction.training.ultralytics_model_context_extractor import (
    UltralyticsModelContextExtractor,
)


@step
def ultralytics_weights_extractor() -> UltralyticsModelContext:
    context: PicselliaTrainingContext = Pipeline.get_active_context()
    model_context_extractor = UltralyticsModelContextExtractor(
        experiment=context.experiment
    )
    model_context = model_context_extractor.get_model_context()
    model_context.download_weights()
    model_context.load_model()
    return model_context
